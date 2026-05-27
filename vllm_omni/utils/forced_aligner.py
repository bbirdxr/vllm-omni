# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared forced aligner utility for streaming TTS word timestamps (issue #3631).

Hosts a single in-process ``vllm.LLM(runner="pooling")`` running upstream's
:class:`vllm.model_executor.models.qwen3_asr_forced_aligner.\
Qwen3ASRForcedAlignerForTokenClassification`. The whole TTS frontend
shares one instance — the aligner is always the slowest path in the
audio request, so a single GPU-resident model is enough.

Public API
----------
* :func:`build_forced_aligner_config` — projects CLI args into a
  ``ForcedAlignerConfig | None``. ``None`` means "feature off".
* :func:`align` — async wrapper around ``llm.encode``; lazy-loads the
  underlying ``vllm.LLM`` on first call. Returns ``list[WordTimestamp]``
  on success, ``[]`` for silence/no aligned tokens, ``None`` when
  alignment failed (the streaming layer maps this to JSON
  ``timestamps: null`` and keeps audio flowing).

Why a single shared utility, not a subprocess
---------------------------------------------
* The model card says ``LLM(runner="pooling")`` is the canonical
  interface; we just consume it.
* ``llm.encode`` is sync + blocking. We wrap it in ``asyncio.to_thread``
  so the event loop stays responsive without spawning a process.
* PR-2 (later, optional) can move the aligner into the vllm-omni stage
  pipeline; the public surface here stays the same.

Failure semantics
-----------------
* On startup failure (model not found, OOM): the first call to
  :func:`align` raises; the streaming layer catches and degrades to
  ``timestamps: null`` for that request, then disables alignment for
  the rest of it. Subsequent requests retry from scratch.
* On per-request failure (decoding error, model spit out empty result):
  ``align`` returns ``None`` (failure) or ``[]`` (silence). The two are
  intentionally distinguishable so clients can tell "no speech" from
  "alignment failed".
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# --- Prompt template ---
# The forced aligner consumes a Qwen3-ASR chat-style prompt with one
# <|timestamp|> marker per word boundary. The exact placeholder names
# are part of the contract with Qwen/Qwen3-ForcedAligner-0.6B's
# tokenizer; verify against the model card before tweaking.
_AUDIO_PLACEHOLDER = "<|audio_start|><|audio_pad|><|audio_end|>"
_TIMESTAMP_TOKEN = "<|timestamp|>"


@dataclass(frozen=True, slots=True)
class WordTimestamp:
    """Internal alignment record. Converted to the pydantic
    :class:`vllm_omni.entrypoints.openai.protocol.audio.WordTimestamp`
    at the HTTP/WebSocket boundary.

    ``confidence`` is reserved for future calibration; currently the
    decoder does not score per-word and leaves it ``None``.
    """

    word: str
    start_ms: int
    end_ms: int
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class ForcedAlignerConfig:
    """Plain-data config built from CLI args at server startup.

    Captures only the fields needed to construct the ``vllm.LLM``
    instance; the LLM itself is lazy-loaded inside :func:`align`.
    Override defaults by setting the matching CLI flag (currently only
    ``--forced-aligner``; the rest follow conservative defaults that
    are enough for Qwen/Qwen3-ForcedAligner-0.6B on a 24GB GPU).
    """

    model: str
    architecture: str = "Qwen3ASRForcedAlignerForTokenClassification"
    pooling_task: str = "token_classify"
    gpu_memory_utilization: float = 0.25
    dtype: str = "bfloat16"
    max_model_len: int = 4096
    trust_remote_code: bool = True


def build_forced_aligner_config(args: Any) -> ForcedAlignerConfig | None:
    """Build a config from CLI args, or ``None`` when the flag is off.

    Mirrors @Dmaner's #3804 helper of the same name so test fixtures
    that target either implementation port unchanged.
    """
    model = getattr(args, "forced_aligner", None)
    if not model:
        return None
    return ForcedAlignerConfig(model=str(model))


# --- Singleton state ---
# A single LLM serves the whole API server. The lock guards the lazy
# constructor; once `_llm` is set, callers can read it lock-free.
_lock = threading.Lock()
_llm: Any = None
_classify_num: int | None = None
_timestamp_token_id: int | None = None
_loaded_config: ForcedAlignerConfig | None = None


async def align(
    *,
    audio: bytes,
    text: str,
    sample_rate: int,
    config: ForcedAlignerConfig,
) -> list[WordTimestamp] | None:
    """Run one forced-alignment pass.

    Args:
        audio: Signed-int16 little-endian mono PCM bytes.
        text: Ground-truth text whose words to align.
        sample_rate: Sample rate of ``audio`` in Hz.
        config: Aligner config (same instance for every call across the
            server's lifetime; reload requires a server restart).

    Returns:
        List of :class:`WordTimestamp` on success (possibly empty for
        silence / no aligned tokens), ``None`` if alignment failed.
    """
    try:
        return await asyncio.to_thread(_align_sync, audio, text, sample_rate, config)
    except Exception:  # noqa: BLE001
        logger.exception("Forced aligner failed for text=%r", text)
        return None


def _align_sync(
    audio: bytes,
    text: str,
    sample_rate: int,
    config: ForcedAlignerConfig,
) -> list[WordTimestamp]:
    _ensure_loaded(config)
    audio_arr = _pcm_bytes_to_float32(audio)
    if audio_arr.size == 0:
        return []
    audio_duration_ms = (audio_arr.size / sample_rate) * 1000.0

    prompt = _build_prompt(text)
    request = {
        "prompt": prompt,
        "multi_modal_data": {"audio": (audio_arr, sample_rate)},
    }

    # Lazy import so ``vllm.pooling_params`` doesn't hit the parent
    # process until alignment is actually invoked.
    from vllm.pooling_params import PoolingParams

    outputs = _llm.encode(  # type: ignore[union-attr]
        [request],
        pooling_params=PoolingParams(),
        pooling_task=config.pooling_task,
        use_tqdm=False,
    )
    if not outputs:
        return []

    result = outputs[0]
    logits = result.outputs.data  # [n_token, classify_num]
    prompt_token_ids = list(result.prompt_token_ids)
    timestamp_positions = [i for i, tid in enumerate(prompt_token_ids) if tid == _timestamp_token_id]
    if not timestamp_positions:
        logger.warning(
            "No <|timestamp|> tokens found in prompt for text=%r; aligner returned %d rows.",
            text,
            logits.shape[0] if hasattr(logits, "shape") else len(logits),
        )
        return []

    return _decode_timestamps(
        logits=logits,
        text=text,
        timestamp_positions=timestamp_positions,
        classify_num=_classify_num,
        audio_duration_ms=audio_duration_ms,
    )


def _ensure_loaded(config: ForcedAlignerConfig) -> None:
    """Lazy-load the singleton ``vllm.LLM`` under lock; idempotent."""
    global _llm, _classify_num, _timestamp_token_id, _loaded_config

    if _llm is not None:
        if _loaded_config is not None and _loaded_config.model != config.model:
            # Multiple configs from different requests — refuse rather
            # than swap models silently. A server restart is required.
            raise RuntimeError(
                f"Forced aligner already loaded with model={_loaded_config.model!r}; "
                f"cannot serve a request that asks for {config.model!r}. "
                "Restart the server to change the aligner model."
            )
        return

    with _lock:
        if _llm is not None:
            return  # raced; another caller did the load

        # Lazy import: vllm pulls torch + CUDA, which we want to avoid
        # at module import time.
        from vllm import LLM

        logger.info(
            "Loading forced aligner %s (gpu_memory_utilization=%.2f)",
            config.model,
            config.gpu_memory_utilization,
        )
        llm = LLM(
            model=config.model,
            runner="pooling",
            hf_overrides={"architectures": [config.architecture]},
            gpu_memory_utilization=config.gpu_memory_utilization,
            trust_remote_code=config.trust_remote_code,
            dtype=config.dtype,
            max_model_len=config.max_model_len,
        )

        thinker_config = getattr(llm.llm_engine.model_config.hf_config, "thinker_config", None)
        if thinker_config is None or not hasattr(thinker_config, "classify_num"):
            raise RuntimeError(
                "Loaded aligner has no thinker_config.classify_num; "
                "expected a Qwen3ASRForcedAlignerForTokenClassification checkpoint."
            )

        tokenizer = llm.get_tokenizer()
        timestamp_token_id = _resolve_timestamp_token_id(tokenizer)

        # Publish in this order so a concurrent reader either sees
        # _llm == None (will block on the lock) or sees a fully
        # initialized aligner.
        _classify_num = int(thinker_config.classify_num)
        _timestamp_token_id = timestamp_token_id
        _loaded_config = config
        _llm = llm

        logger.info(
            "Forced aligner ready: timestamp_token_id=%d, classify_num=%d",
            timestamp_token_id,
            _classify_num,
        )


# --- pure helpers (testable without a GPU / vllm) ---


def _build_prompt(text: str) -> str:
    """Construct the chat-style prompt with per-word timestamp markers."""
    words = text.split()
    if not words:
        # Pad with one timestamp so the decoder always has something to
        # read; an empty result still surfaces as "[]" upstream.
        body = _TIMESTAMP_TOKEN
    else:
        body = " ".join(f"{w} {_TIMESTAMP_TOKEN}" for w in words)
    return f"<|im_start|>user\n{_AUDIO_PLACEHOLDER}{body}<|im_end|>\n<|im_start|>assistant\n"


def _resolve_timestamp_token_id(tokenizer: Any) -> int:
    """Look up the integer id of the timestamp special token."""
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        raise RuntimeError("Aligner tokenizer has no convert_tokens_to_ids method.")
    tid = convert(_TIMESTAMP_TOKEN)
    if isinstance(tid, list):
        tid = tid[0] if tid else None
    if tid is None or (isinstance(tid, int) and tid < 0):
        raise RuntimeError(
            f"Aligner tokenizer does not recognise {_TIMESTAMP_TOKEN!r} (got id={tid}). "
            "Check the model card; the marker token may use a different name."
        )
    return int(tid)


def _pcm_bytes_to_float32(audio: bytes) -> np.ndarray:
    """Decode signed-int16 mono PCM bytes into a [-1, 1] float32 array."""
    if not audio:
        return np.zeros(0, dtype=np.float32)
    if len(audio) % 2 != 0:
        # Drop a trailing odd byte rather than raise; keeps streaming
        # robust against off-by-one chunk boundaries.
        audio = audio[:-1]
    pcm = np.frombuffer(audio, dtype=np.int16)
    return (pcm.astype(np.float32) / 32768.0).copy()


def _decode_timestamps(
    *,
    logits: Any,
    text: str,
    timestamp_positions: list[int],
    classify_num: int,
    audio_duration_ms: float,
) -> list[WordTimestamp]:
    """Translate ``[n_token, classify_num]`` logits into word timestamps."""
    arr = logits.detach().cpu().numpy() if hasattr(logits, "detach") else np.asarray(logits)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D logits [n_token, classify_num]; got shape {arr.shape}")
    if arr.shape[1] != classify_num:
        raise ValueError(
            f"Logits last dim {arr.shape[1]} != classify_num {classify_num}; "
            "model config and prompt template may be out of sync."
        )

    words = text.split()
    expected = len(words) + 1
    if len(timestamp_positions) != expected:
        logger.warning(
            "Got %d timestamp positions but text has %d words (expected %d markers); returning empty alignment.",
            len(timestamp_positions),
            len(words),
            expected,
        )
        return []

    marker_logits = arr[timestamp_positions, :]
    bin_idx = marker_logits.argmax(axis=-1)
    bin_size_ms = audio_duration_ms / classify_num if classify_num > 0 else 0.0

    out: list[WordTimestamp] = []
    for i, word in enumerate(words):
        start_bin = int(bin_idx[i])
        end_bin = int(bin_idx[i + 1])
        if end_bin < start_bin:
            # Pathological output; skip this word rather than crash.
            continue
        out.append(
            WordTimestamp(
                word=word,
                start_ms=int(round(start_bin * bin_size_ms)),
                end_ms=int(round(end_bin * bin_size_ms)),
                confidence=None,
            )
        )
    return out


# Test hooks ---------------------------------------------------------------
# Tests need a way to reset module state without restarting Python. Not
# part of the public API; do not call in production code.


def _reset_for_tests() -> None:
    """Drop the cached aligner state so the next call reloads."""
    global _llm, _classify_num, _timestamp_token_id, _loaded_config
    with _lock:
        _llm = None
        _classify_num = None
        _timestamp_token_id = None
        _loaded_config = None
