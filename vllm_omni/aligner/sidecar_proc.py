# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Aligner sidecar subprocess entry point.

Owns the forced-aligner model in its own Python process so it cannot
interfere with the API server's event loop or the vLLM workers' GPU
memory accounting. PR-2 replaces this entry with a native stage worker
without changing the queue contract.

Lifecycle:

1. Parent constructs ``in_q``, ``out_q``, ``ready_q`` (multiprocessing
   queues) and spawns this function via ``mp.Process``.
2. Child sets ``CUDA_VISIBLE_DEVICES`` from ``args.aligner_device`` so
   the GPU index lookup matches the parent's view.
3. Child loads the aligner model + processor lazily, calls
   ``torch.cuda.set_per_process_memory_fraction(args.aligner_gpu_memory)``
   to keep its slice off the vLLM workers' books, then puts a single
   ``"READY"`` token onto ``ready_q``.
4. Busy loop: ``in_q.get()`` -> forward -> ``out_q.put(AlignmentResponse)``.
5. ``SHUTDOWN_SIGNAL`` on ``in_q`` (or process signal) breaks the loop.

If anything in steps 2-3 fails, the child puts the exception text onto
``ready_q`` and exits. The parent treats this as an aligner-startup
failure and refuses to bring the API server up.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import traceback
from dataclasses import dataclass
from multiprocessing.queues import Queue as MPQueue

from vllm_omni.aligner.types import (
    SHUTDOWN_SIGNAL,
    AlignmentRequest,
    AlignmentResponse,
)

logger = logging.getLogger(__name__)

_READY_TOKEN = "READY"

_ASCII_BANNER = (
    "Aligner sidecar (issue #3631) starting; this process owns the forced "
    "aligner model and communicates with the API server via mp.Queue."
)


@dataclass(frozen=True, slots=True)
class SidecarArgs:
    """Plain-data subset of ``OmniEngineArgs`` passed across the process boundary.

    Kept narrow on purpose: the sidecar must not touch the vLLM config
    object, the engine's parallel state, or anything that would force it
    to reproduce the parent's complex import graph.
    """

    aligner_model: str
    aligner_gpu_memory: float
    aligner_device: str

    @classmethod
    def from_engine_args(cls, engine_args) -> SidecarArgs:
        """Project the relevant fields out of ``OmniEngineArgs`` (or a duck type)."""
        return cls(
            aligner_model=engine_args.aligner_model,
            aligner_gpu_memory=float(engine_args.aligner_gpu_memory),
            aligner_device=engine_args.aligner_device,
        )


def aligner_sidecar_main(
    in_q: MPQueue,
    out_q: MPQueue,
    ready_q: MPQueue,
    args: SidecarArgs,
) -> None:
    """Subprocess entry. Must be picklable for ``mp.Process``."""
    # Reset Python signal handlers: the parent's handlers (uvicorn's
    # graceful-shutdown hooks) are inherited by default which leaks state.
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)

    try:
        runtime = _AlignerRuntime.startup(args)
    except Exception as exc:
        # Carry the traceback to the parent so the API server can show
        # the user a real reason for refusing to start.
        ready_q.put(("ERROR", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))
        return

    ready_q.put((_READY_TOKEN, None))
    logger.info(_ASCII_BANNER)

    try:
        _busy_loop(runtime, in_q, out_q)
    except KeyboardInterrupt:
        logger.info("Aligner sidecar received KeyboardInterrupt; exiting.")
    finally:
        runtime.shutdown()


def _busy_loop(runtime: _AlignerRuntime, in_q: MPQueue, out_q: MPQueue) -> None:
    """Pop requests, run alignment, push responses; honor SHUTDOWN_SIGNAL."""
    while True:
        msg = in_q.get()  # blocks; exit only via SHUTDOWN_SIGNAL or signal
        if msg is SHUTDOWN_SIGNAL or type(msg).__name__ == "_ShutdownSignal":
            return
        if not isinstance(msg, AlignmentRequest):
            logger.warning("Aligner sidecar received unknown message type: %s", type(msg).__name__)
            continue

        try:
            timestamps = runtime.align(
                audio=msg.audio,
                text=msg.text,
                sample_rate=msg.sample_rate,
            )
            out_q.put(
                AlignmentResponse(
                    req_id=msg.req_id,
                    sentence_index=msg.sentence_index,
                    chunk_id=msg.chunk_id,
                    ok=True,
                    timestamps=timestamps,
                )
            )
        except Exception as exc:
            # Single-chunk failure must not bring the sidecar down: emit
            # an ok=False response and keep serving subsequent chunks.
            logger.exception("Aligner forward failed for req=%s chunk=%s", msg.req_id, msg.chunk_id)
            out_q.put(
                AlignmentResponse(
                    req_id=msg.req_id,
                    sentence_index=msg.sentence_index,
                    chunk_id=msg.chunk_id,
                    ok=False,
                    timestamps=[],
                    error=f"{type(exc).__name__}: {exc}",
                )
            )


class _AlignerRuntime:
    """Holds the loaded aligner model + processor inside the subprocess.

    The actual ``align()`` implementation is intentionally a thin shim
    around :mod:`vllm_omni.aligner.ctc_decode` so the model-loading
    plumbing stays separate from the alignment math (and so the math
    can be unit-tested without spawning a process).
    """

    def __init__(self, model: object, processor: object, frame_hop_ms: float):
        self._model = model
        self._processor = processor
        self._frame_hop_ms = frame_hop_ms

    @classmethod
    def startup(cls, args: SidecarArgs) -> _AlignerRuntime:
        # Restrict CUDA visibility before importing torch so the device
        # index the rest of the sidecar uses (always cuda:0 internally)
        # maps to the user-requested physical GPU.
        if args.aligner_device.startswith("cuda:"):
            os.environ["CUDA_VISIBLE_DEVICES"] = args.aligner_device.split(":", 1)[1]

        # Lazy imports: torch / transformers must not enter the parent
        # process even when --enable-word-timestamps is off.
        import torch
        from transformers import AutoModelForCTC, AutoProcessor

        if torch.cuda.is_available() and args.aligner_device.startswith("cuda:"):
            torch.cuda.set_per_process_memory_fraction(args.aligner_gpu_memory, device=0)

        logger.info("Loading aligner model %s on %s", args.aligner_model, args.aligner_device)
        processor = AutoProcessor.from_pretrained(args.aligner_model)
        model = AutoModelForCTC.from_pretrained(
            args.aligner_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        if args.aligner_device.startswith("cuda"):
            model = model.cuda()
        elif args.aligner_device.startswith(("npu:", "xpu:")):
            # Best-effort device move; .to() raises a clear error if the
            # backend isn't installed in this environment.
            model = model.to(args.aligner_device)
        model.eval()

        # Frame-rate / hop is needed to convert CTC frame indices to ms.
        # Different aligners expose this differently; ctc_decode resolves
        # it from the model config.
        from vllm_omni.aligner.ctc_decode import resolve_frame_hop_ms

        frame_hop_ms = resolve_frame_hop_ms(model.config)
        return cls(model=model, processor=processor, frame_hop_ms=frame_hop_ms)

    def align(self, audio: bytes, text: str, sample_rate: int) -> list[dict]:
        """Forward pass + CTC decode for one chunk.

        Returns a list of dicts with keys ``word``, ``start_ms``,
        ``end_ms``, ``confidence``. Empty list is a valid result
        (silence / no aligned tokens).
        """
        # Lazy imports keep the sidecar's startup deterministic even when
        # this method is patched out in tests.
        import torch

        from vllm_omni.aligner.ctc_decode import (
            decode_ctc_alignment,
            pcm_bytes_to_float_array,
        )

        audio_array = pcm_bytes_to_float_array(audio, sample_rate=sample_rate)
        with torch.inference_mode():
            inputs = self._processor(
                audio=audio_array,
                sampling_rate=sample_rate,
                return_tensors="pt",
            )
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            logits = self._model(**inputs).logits  # [1, T, vocab_size]

        return decode_ctc_alignment(
            logits=logits[0].float().cpu(),
            text=text,
            processor=self._processor,
            frame_hop_ms=self._frame_hop_ms,
        )

    def shutdown(self) -> None:
        """Best-effort cleanup. No-op for plain transformers models."""
        # Hold onto the model reference so any in-flight CUDA frees can
        # finish; the OS will reclaim the rest at process exit.
        self._model = None
        self._processor = None
        sys.stdout.flush()
        sys.stderr.flush()
