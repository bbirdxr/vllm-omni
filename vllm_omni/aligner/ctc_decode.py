# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CTC + char-to-word post-processing for the forced aligner.

Pure functions, no GPU side effects, no hidden state. The sidecar calls
into here after running the aligner forward; tests can exercise these
functions directly with a synthetic logits tensor.

Algorithm overview (greedy CTC, single chunk):

1. ``argmax`` over the time axis of logits -> per-frame token id.
2. Collapse runs of the same id, drop the CTC blank id. Each surviving
   run becomes one ``(token, start_frame, end_frame)`` triple.
3. Decode token ids back to characters / sub-words via the aligner's
   tokenizer.
4. Convert frame indices to milliseconds using ``frame_hop_ms``.
5. Optional CJK collapse: keep characters as separate timestamps; the
   client can join them into words with its preferred segmentation.

Forced-alignment refinement (Viterbi against the ground-truth ``text``)
is left as a follow-up — greedy CTC works well enough on the model's
own output to ship the contract layer. See ``_TODO_FORCED_ALIGNMENT``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Default for wav2vec2-style CTC heads at 16 kHz: 20 ms per logit frame
# (= 320 sample stride / 16 kHz). Qwen3-ForcedAligner-0.6B reuses this
# convention according to its model card; verified at integration time
# in ``resolve_frame_hop_ms``.
_DEFAULT_FRAME_HOP_MS = 20.0

# Marker for follow-up work; not a runtime concern.
_TODO_FORCED_ALIGNMENT = (
    "PR-1 ships greedy CTC. PR-2 (or a follow-up) should add Viterbi "
    "alignment against the ground-truth text for tighter boundaries on "
    "ambiguous frames."
)


def pcm_bytes_to_float_array(audio: bytes, sample_rate: int) -> np.ndarray:
    """Decode signed-int16 mono PCM bytes into a [-1, 1] float32 array.

    The TTS pipeline emits signed-int16 little-endian by convention;
    callers that need a different format should convert before submitting.
    """
    if not audio:
        return np.zeros(0, dtype=np.float32)
    if len(audio) % 2 != 0:
        # Drop a trailing odd byte rather than raise; keeps streaming
        # robust against off-by-one chunk boundaries.
        audio = audio[:-1]
    pcm = np.frombuffer(audio, dtype=np.int16)
    return (pcm.astype(np.float32) / 32768.0).copy()


def resolve_frame_hop_ms(config: Any) -> float:
    """Best-effort lookup of milliseconds per CTC logit frame from a model config.

    Order of preference, all optional:

    1. Explicit ``frame_hop_ms`` attribute (custom Qwen3-Aligner addition).
    2. ``conv_stride`` * ``conv_kernel`` math from wav2vec2 / Hubert
       config combined with ``inputs_to_logits_ratio`` if exposed.
    3. ``inputs_to_logits_ratio`` directly (preferred by HF for newer CTC
       models): ``ratio / sampling_rate * 1000``.
    4. Hard-coded 20 ms wav2vec2 default with a warning.

    Returns:
        Milliseconds per frame as a float. Always > 0.
    """
    explicit = getattr(config, "frame_hop_ms", None)
    if isinstance(explicit, (int, float)) and explicit > 0:
        return float(explicit)

    sampling_rate = getattr(config, "sampling_rate", None)
    ratio = getattr(config, "inputs_to_logits_ratio", None)
    if sampling_rate and ratio:
        return (float(ratio) / float(sampling_rate)) * 1000.0

    conv_strides = getattr(config, "conv_stride", None)
    if conv_strides and sampling_rate:
        # Product of all conv strides == samples per logit frame.
        try:
            stride_product = 1
            for s in conv_strides:
                stride_product *= int(s)
            return (stride_product / float(sampling_rate)) * 1000.0
        except (TypeError, ValueError):
            pass

    logger.warning(
        "Could not resolve frame hop from model config; falling back to "
        "%.1f ms (wav2vec2 default). If timestamps look stretched/squashed, "
        "set frame_hop_ms on the model config explicitly.",
        _DEFAULT_FRAME_HOP_MS,
    )
    return _DEFAULT_FRAME_HOP_MS


def decode_ctc_alignment(
    logits: np.ndarray | Any,
    text: str,
    processor: Any,
    frame_hop_ms: float,
) -> list[dict]:
    """Greedy CTC decode + frame->ms conversion.

    Args:
        logits: ``[T, vocab_size]`` tensor or ndarray of pre-softmax logits
            for one chunk. ``argmax`` is taken along the last axis.
        text: Ground-truth text for the chunk (currently informational
            only; greedy CTC ignores it but the parameter is kept stable
            for future Viterbi forced alignment).
        processor: Aligner ``processor`` / tokenizer used to decode token
            ids back to characters. Must expose ``tokenizer.pad_token_id``
            (interpreted as the CTC blank) and either
            ``tokenizer.decode`` or ``tokenizer.convert_ids_to_tokens``.
        frame_hop_ms: Milliseconds per logit frame.

    Returns:
        A list of dicts ``{word, start_ms, end_ms, confidence}``.
        ``confidence`` is the mean softmax probability of the argmax id
        over the frames spanned by that token; ``None`` when the
        computation cannot be performed (e.g. the logits backend doesn't
        expose softmax cheaply).
    """
    # Tolerate both numpy arrays and torch tensors without forcing a
    # torch dependency in unit tests.
    arr = _to_numpy(logits)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D logits [T, V]; got shape {arr.shape}")
    if arr.size == 0:
        return []

    blank_id = _resolve_blank_id(processor)
    ids = arr.argmax(axis=-1)  # [T]

    # Collapse runs and drop blanks; track frame spans for ms conversion.
    runs: list[tuple[int, int, int]] = []  # (token_id, start_frame, end_frame)
    cur_id = -1
    cur_start = 0
    for t, tok in enumerate(ids):
        tok = int(tok)
        if tok == cur_id:
            continue
        if cur_id != -1 and cur_id != blank_id:
            runs.append((cur_id, cur_start, t))  # half-open end
        cur_id = tok
        cur_start = t
    # Close the trailing run.
    if cur_id != -1 and cur_id != blank_id:
        runs.append((cur_id, cur_start, len(ids)))

    if not runs:
        return []

    # Decode token ids to characters in batch where possible.
    token_ids = [r[0] for r in runs]
    tokens = _decode_token_ids(processor, token_ids)

    # Per-token mean confidence (softmax of the argmax id over the run).
    confidences = _per_token_confidence(arr, runs)

    out: list[dict] = []
    for (tok_id, start_f, end_f), tok_str, conf in zip(runs, tokens, confidences, strict=False):
        if not tok_str or tok_str.strip() == "":
            continue  # Skip whitespace tokens introduced by some tokenizers.
        out.append(
            {
                "word": tok_str,
                "start_ms": int(round(start_f * frame_hop_ms)),
                "end_ms": int(round(end_f * frame_hop_ms)),
                "confidence": conf,
            }
        )
    return out


# --------- internal helpers ---------


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    # Avoid importing torch at module level so the API server can import
    # this module without torch in PR-1's failure-recovery path.
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _resolve_blank_id(processor: Any) -> int:
    """Look up the CTC blank id (= padding token id by HF convention)."""
    tok = getattr(processor, "tokenizer", processor)
    blank_id = getattr(tok, "pad_token_id", None)
    if blank_id is not None:
        return int(blank_id)
    # Fall back to id 0; almost all wav2vec2-family CTC heads put blank
    # at index 0. Logged so misconfiguration is visible.
    logger.warning("Aligner processor exposes no pad_token_id; assuming CTC blank id = 0.")
    return 0


def _decode_token_ids(processor: Any, token_ids: list[int]) -> list[str]:
    """Convert token ids -> string tokens, robust to both fast and slow tokenizers."""
    tok = getattr(processor, "tokenizer", processor)
    convert = getattr(tok, "convert_ids_to_tokens", None)
    if callable(convert):
        result = convert(token_ids)
        if isinstance(result, str):
            return [result]
        return [_strip_subword_marker(s) for s in result]

    decode = getattr(tok, "decode", None)
    if callable(decode):
        # Decoding one-by-one to preserve per-frame alignment with runs.
        return [_strip_subword_marker(decode([tid])) for tid in token_ids]

    raise RuntimeError(
        "Aligner processor exposes neither convert_ids_to_tokens nor decode; cannot turn token ids into strings."
    )


def _strip_subword_marker(s: str) -> str:
    """Drop the leading SentencePiece/WordPiece marker for word-piece tokenizers.

    SentencePiece uses U+2581 (▁), WordPiece uses ``##``. For CJK these
    rarely appear, but staying lenient keeps the function model-agnostic.
    """
    if s.startswith("\u2581"):
        return s[1:]
    if s.startswith("##"):
        return s[2:]
    return s


def _per_token_confidence(
    logits: np.ndarray,
    runs: list[tuple[int, int, int]],
) -> list[float | None]:
    """Mean softmax probability of the argmax token across each run's frames."""
    if logits.size == 0:
        return [None] * len(runs)
    # Numerically stable softmax along the vocab axis.
    shifted = logits - logits.max(axis=-1, keepdims=True)
    np.exp(shifted, out=shifted)
    shifted /= shifted.sum(axis=-1, keepdims=True)
    out: list[float | None] = []
    for tok_id, start, end in runs:
        if end <= start:
            out.append(None)
            continue
        probs = shifted[start:end, tok_id]
        out.append(float(probs.mean()))
    return out
