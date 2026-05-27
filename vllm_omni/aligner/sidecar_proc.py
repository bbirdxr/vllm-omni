# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Aligner sidecar subprocess entry point.

Owns a standalone ``vllm.LLM`` instance running the upstream
:class:`vllm.model_executor.models.qwen3_asr_forced_aligner.\
Qwen3ASRForcedAlignerForTokenClassification` model. The instance lives
in its own Python process so its CUDA context, GPU memory, and pooling
runner do not interfere with the TTS workers' generation engine.

PR-2 replaces this entry with a native ``LLM_GENERATION``-style stage
worker (or a new ``StageExecutionType.POOLING`` if the maintainers
prefer); either way the queue contract on top of it stays unchanged.

Lifecycle:

1. Parent constructs ``in_q``, ``out_q``, ``ready_q`` (multiprocessing
   queues) and spawns this function via ``mp.Process``.
2. Child sets ``CUDA_VISIBLE_DEVICES`` from ``args.aligner_device`` so
   the visible-index inside the child is always 0.
3. Child constructs a ``vllm.LLM`` with ``runner="pooling"`` and the
   ``Qwen3ASRForcedAlignerForTokenClassification`` architecture
   override, then puts a single ``"READY"`` token onto ``ready_q``.
4. Busy loop: ``in_q.get()`` -> ``llm.encode`` ->
   ``decode_alignment_outputs`` -> ``out_q.put(AlignmentResponse)``.
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
    """Holds a standalone ``vllm.LLM`` instance inside the subprocess.

    Implementation is deliberately thin: the upstream
    :class:`Qwen3ASRForcedAlignerForTokenClassification` model already
    ships with the multimodal processor, classifier head, and
    ``token_classify`` pooler. This class is just the sidecar-side
    adapter for the queue contract.
    """

    # Pulled out as class constants so tests can override them and so
    # the values are visible at the top of the file (vs buried inside
    # the LLM(...) call).
    _POOLING_TASK = "token_classify"
    _ARCHITECTURE = "Qwen3ASRForcedAlignerForTokenClassification"

    def __init__(
        self,
        llm: object,
        timestamp_token_id: int,
        classify_num: int,
    ) -> None:
        self._llm = llm
        self._timestamp_token_id = timestamp_token_id
        self._classify_num = classify_num

    @classmethod
    def startup(cls, args: SidecarArgs) -> _AlignerRuntime:
        # Restrict CUDA visibility before importing vllm so the
        # in-process device index (always 0 from here on) maps to the
        # user-requested physical GPU.
        if args.aligner_device.startswith("cuda:"):
            os.environ["CUDA_VISIBLE_DEVICES"] = args.aligner_device.split(":", 1)[1]

        # Lazy import: vllm must not enter the parent process even when
        # --enable-word-timestamps is off, otherwise we double-import
        # CUDA contexts.
        from vllm import LLM

        logger.info(
            "Loading aligner %s via vllm pooling runner on %s",
            args.aligner_model,
            args.aligner_device,
        )
        llm = LLM(
            model=args.aligner_model,
            runner="pooling",
            hf_overrides={"architectures": [cls._ARCHITECTURE]},
            gpu_memory_utilization=args.aligner_gpu_memory,
            trust_remote_code=True,
            enforce_eager=False,
        )

        # Resolve the two model-derived constants the decoder needs:
        #   * timestamp special-token id (locates marker rows in the
        #     [n_token, classify_num] logits tensor returned by the
        #     pooler)
        #   * classify_num (number of time bins per timestamp slot)
        from vllm_omni.aligner.qwen3_aligner import resolve_timestamp_token_id

        tokenizer = llm.get_tokenizer()
        timestamp_token_id = resolve_timestamp_token_id(tokenizer)

        thinker_config = getattr(llm.llm_engine.model_config.hf_config, "thinker_config", None)
        if thinker_config is None or not hasattr(thinker_config, "classify_num"):
            raise RuntimeError(
                "Loaded aligner model has no thinker_config.classify_num; "
                "expected a Qwen3ASRForcedAlignerForTokenClassification checkpoint."
            )
        classify_num = int(thinker_config.classify_num)

        logger.info(
            "Aligner ready: timestamp_token_id=%d, classify_num=%d",
            timestamp_token_id,
            classify_num,
        )
        return cls(llm=llm, timestamp_token_id=timestamp_token_id, classify_num=classify_num)

    def align(self, audio: bytes, text: str, sample_rate: int) -> list[dict]:
        """Run one forced-alignment pass; return word-timestamp dicts.

        The empty list is a valid result (no aligned tokens — e.g. a
        silence chunk). Failures raise; the caller turns those into
        ``AlignmentResponse(ok=False, ...)`` so the streaming layer
        emits ``timestamps: null``.
        """
        from vllm.pooling_params import PoolingParams

        from vllm_omni.aligner.qwen3_aligner import (
            build_aligner_prompt,
            decode_alignment_outputs,
            find_timestamp_positions,
            pcm_bytes_to_float_array,
        )

        audio_array = pcm_bytes_to_float_array(audio)
        if audio_array.size == 0:
            return []
        audio_duration_ms = (audio_array.size / sample_rate) * 1000.0

        prompt = build_aligner_prompt(text, sample_rate=sample_rate)
        request = {
            "prompt": prompt,
            "multi_modal_data": {"audio": (audio_array, sample_rate)},
        }
        outputs = self._llm.encode(  # type: ignore[union-attr]
            [request],
            pooling_params=PoolingParams(),
            pooling_task=self._POOLING_TASK,
            use_tqdm=False,
        )
        if not outputs:
            return []

        result = outputs[0]
        # PoolingRequestOutput.outputs.data is [n_token, classify_num].
        logits = result.outputs.data
        prompt_token_ids = list(result.prompt_token_ids)
        timestamp_positions = find_timestamp_positions(prompt_token_ids, self._timestamp_token_id)
        if not timestamp_positions:
            logger.warning(
                "No <|timestamp|> tokens found in prompt for text=%r; aligner returned %d rows.",
                text,
                logits.shape[0],
            )
            return []

        return decode_alignment_outputs(
            logits=logits,
            text=text,
            timestamp_positions=timestamp_positions,
            classify_num=self._classify_num,
            audio_duration_ms=audio_duration_ms,
        )

    def shutdown(self) -> None:
        """Best-effort cleanup; the OS reclaims CUDA at process exit."""
        # Drop the LLM ref so __del__ can run before the process exits;
        # any in-flight CUDA frees finish on the way out.
        self._llm = None
        sys.stdout.flush()
        sys.stderr.flush()
