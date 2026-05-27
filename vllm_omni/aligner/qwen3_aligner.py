# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapter for Qwen3-ASR-ForcedAligner via vLLM's pooling runner.

This module is the thin layer between our sidecar's mp.Queue interface
and vLLM's existing
:class:`vllm.model_executor.models.qwen3_asr_forced_aligner.Qwen3ASRForcedAlignerForTokenClassification`.

Why this exists
---------------
Upstream vLLM already implements the model end-to-end as a pooling
model with ``pooling_task="token_classify"`` (see the model docstring
for the canonical usage pattern). Our sidecar's job is therefore not to
re-implement the alignment math but to:

1. Construct the per-request prompt that mixes the audio placeholder
   tokens with text and ``<timestamp>`` markers, in the format the
   upstream multimodal processor expects.
2. Call ``llm.encode(..., pooling_task="token_classify")``.
3. Translate the returned ``[n_token, classify_num]`` logits back into
   a flat list of ``(word, start_ms, end_ms, confidence)`` dicts
   suitable for streaming back to the API client.

What's still TODO
-----------------
The exact prompt template (which audio placeholder tokens to splice in,
which delimiter goes between text words and ``<timestamp>`` markers,
how the model expects the timestamp slots to be tokenized) is fixed by
the model card. We've left the template in
:func:`build_aligner_prompt` parameterised against a small set of
constants documented near the top of the function so it can be
calibrated once with a working checkpoint without having to touch the
sidecar code.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

# --- Prompt template (Qwen3-ASR family, calibrated against the model card) ---
# These constants live here, not on the model config, because they are
# part of the *prompt contract* between vllm-omni and Qwen3-ForcedAligner;
# changing them is changing how we ask the model, not how the model
# computes. See the upstream model docstring at
# vllm/model_executor/models/qwen3_asr_forced_aligner.py for canonical
# usage and confirm before tweaking.
_AUDIO_PLACEHOLDER = "<|audio_start|><|audio_pad|><|audio_end|>"
_TIMESTAMP_TOKEN = "<|timestamp|>"
_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"


def pcm_bytes_to_float_array(audio: bytes) -> np.ndarray:
    """Decode signed-int16 mono PCM bytes into a [-1, 1] float32 array.

    The TTS pipeline emits signed-int16 little-endian by convention;
    the aligner's audio tower expects float in [-1, 1].
    """
    if not audio:
        return np.zeros(0, dtype=np.float32)
    if len(audio) % 2 != 0:
        # Drop a trailing odd byte rather than raise; keeps streaming
        # robust against off-by-one chunk boundaries.
        audio = audio[:-1]
    pcm = np.frombuffer(audio, dtype=np.int16)
    return (pcm.astype(np.float32) / 32768.0).copy()


def build_aligner_prompt(text: str, *, sample_rate: int) -> str:
    """Build the chat-template prompt for a single forced-alignment request.

    Args:
        text: Ground-truth text whose tokens should be aligned.
        sample_rate: Sample rate of the audio chunk; included for parity
            with upstream's STT prompt builder even though the audio
            tensor is supplied separately via ``multi_modal_data``.

    Returns:
        A string prompt ready to be paired with
        ``multi_modal_data={"audio": <ndarray>}`` and passed to
        ``llm.encode(...)``.

    Notes:
        Each whitespace-separated word in ``text`` is followed by a
        ``<|timestamp|>`` marker. At inference time, the model emits a
        per-token classification probability over time bins; we read off
        only the marker positions to produce ``(word, start_ms, end_ms)``
        triples (see :func:`decode_alignment_outputs`).

        For CJK text without spaces we still rely on whitespace splits
        — callers should pre-tokenize CJK input with a spaces-between-
        characters convention before submitting (the streaming layer's
        ``--cjk-mode`` switch handles this in PR-1; default keeps
        whitespace as-is).
    """
    del sample_rate  # currently informational; kept for forward compat
    # Word-with-timestamp body: "word1 <|timestamp|> word2 <|timestamp|> ..."
    words = text.split()
    body = " ".join(f"{w} {_TIMESTAMP_TOKEN}" for w in words) if words else _TIMESTAMP_TOKEN
    return f"{_IM_START}user\n{_AUDIO_PLACEHOLDER}{body}{_IM_END}\n{_IM_START}assistant\n"


def decode_alignment_outputs(
    logits: torch.Tensor | np.ndarray,
    *,
    text: str,
    timestamp_positions: list[int],
    classify_num: int,
    audio_duration_ms: float,
) -> list[dict]:
    """Translate token-classify logits into per-word timestamp dicts.

    Args:
        logits: ``[n_token, classify_num]`` activation output of the
            forced-aligner pooler. Either a torch tensor or a numpy
            array — both are accepted to keep tests backend-free.
        text: The same ground-truth text that ``build_aligner_prompt``
            was given, used to label each emitted timestamp.
        timestamp_positions: Indices into ``logits`` of the
            ``<|timestamp|>`` tokens, in left-to-right order. The number
            of positions must match ``len(text.split()) + 1`` (one
            timestamp per word boundary; first/last bracket the chunk).
        classify_num: Number of time bins the classifier head outputs.
            Pulled from ``thinker_config.classify_num`` of the model.
        audio_duration_ms: Duration of the audio chunk in milliseconds;
            used to convert bin indices into absolute milliseconds.

    Returns:
        List of dicts ``{word, start_ms, end_ms, confidence}``. The list
        may be empty (silence / no aligned tokens). ``confidence`` is
        the softmax probability of the chosen bin (None if unavailable).
    """
    arr = _to_numpy(logits)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D logits [n_token, classify_num]; got shape {arr.shape}")
    if arr.shape[1] != classify_num:
        raise ValueError(
            f"Logits last dim {arr.shape[1]} != classify_num {classify_num}; "
            "model config and prompt template may be out of sync."
        )
    if not timestamp_positions:
        return []

    words = text.split()
    n_expected = len(words) + 1
    if len(timestamp_positions) != n_expected:
        # Hard mismatch usually means the prompt template emitted a
        # different number of <|timestamp|> tokens than expected.
        # Surface it loudly so the model card calibration step is
        # forced to happen instead of silently producing wrong times.
        raise ValueError(
            f"Got {len(timestamp_positions)} timestamp positions but "
            f"text has {len(words)} words (expected {n_expected} markers). "
            "Check prompt template against the model card."
        )

    # Pull out only the marker rows: [n_markers, classify_num].
    marker_logits = arr[timestamp_positions, :]
    bin_idx = marker_logits.argmax(axis=-1)  # [n_markers]
    confidences = _softmax_max(marker_logits)  # [n_markers]

    bin_size_ms = audio_duration_ms / classify_num if classify_num > 0 else 0.0

    out: list[dict] = []
    for i, word in enumerate(words):
        start_bin = int(bin_idx[i])
        end_bin = int(bin_idx[i + 1])
        if end_bin < start_bin:
            # Pathological output: skip rather than crash; leave to the
            # caller to log.
            continue
        out.append(
            {
                "word": word,
                "start_ms": int(round(start_bin * bin_size_ms)),
                "end_ms": int(round(end_bin * bin_size_ms)),
                "confidence": float((confidences[i] + confidences[i + 1]) / 2.0),
            }
        )
    return out


def find_timestamp_positions(prompt_token_ids: list[int], timestamp_token_id: int) -> list[int]:
    """Return left-to-right indices of ``<|timestamp|>`` in the tokenized prompt.

    Used by the sidecar after ``llm.encode`` returns; the
    ``PoolingRequestOutput`` carries ``prompt_token_ids`` so we can find
    the marker rows in ``logits`` without re-tokenizing.
    """
    return [i for i, tid in enumerate(prompt_token_ids) if tid == timestamp_token_id]


def resolve_timestamp_token_id(processor: Any) -> int:
    """Look up the integer id of ``<|timestamp|>`` from the aligner tokenizer."""
    tokenizer = getattr(processor, "tokenizer", processor)
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        raise RuntimeError("Aligner tokenizer has no convert_tokens_to_ids method.")
    tid = convert(_TIMESTAMP_TOKEN)
    if isinstance(tid, list):
        if not tid:
            raise RuntimeError(f"Tokenizer returned empty id for {_TIMESTAMP_TOKEN!r}.")
        tid = tid[0]
    if tid is None or (isinstance(tid, int) and tid < 0):
        raise RuntimeError(
            f"Aligner tokenizer does not recognise {_TIMESTAMP_TOKEN!r} "
            f"(got id={tid}). The model checkpoint may be missing the "
            "timestamp special token."
        )
    return int(tid)


# --------- internal helpers ---------


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _softmax_max(logits: np.ndarray) -> np.ndarray:
    """Per-row softmax max — i.e. the chosen bin's probability."""
    if logits.size == 0:
        return np.zeros(logits.shape[0], dtype=np.float32)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=-1, keepdims=True)
    return probs.max(axis=-1)
