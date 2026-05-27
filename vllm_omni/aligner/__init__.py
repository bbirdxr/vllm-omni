# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared forced aligner for TTS streaming word-level timestamps.

This package provides a model-agnostic forced aligner that runs alongside
TTS generation and emits ``(word, start_ms, end_ms)`` triples per audio
chunk. Default-off; opt-in via ``--enable-word-timestamps`` server flag
plus ``word_timestamps=true`` per request.

The sidecar in PR-1 hosts a standalone ``vllm.LLM(runner="pooling")``
running upstream's
:class:`vllm.model_executor.models.qwen3_asr_forced_aligner.\
Qwen3ASRForcedAlignerForTokenClassification`. We do not re-implement the
alignment math; we only build the per-chunk prompt, call ``encode``, and
translate the returned token-classify logits back into word timestamps.

Layout:
    types.py           - data classes shared by the sidecar and its client
    qwen3_aligner.py   - prompt template + token-classify output decoding
    sidecar_proc.py    - subprocess entry that hosts the vllm.LLM instance
    sidecar_client.py  - in-process client managing the subprocess lifecycle

PR-1 wires the sidecar into ``serving_speech_stream.py`` directly.
PR-2 may replace the subprocess with a native pooling stage in the
vllm-omni engine, but the public protocol and CLI flags stay frozen.
"""

from vllm_omni.aligner.types import (
    SHUTDOWN_SIGNAL,
    AlignerError,
    AlignerErrorKind,
    AlignmentRequest,
    AlignmentResponse,
)

__all__ = [
    "SHUTDOWN_SIGNAL",
    "AlignerError",
    "AlignerErrorKind",
    "AlignmentRequest",
    "AlignmentResponse",
]
