# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared forced aligner for TTS streaming word-level timestamps.

This package provides a model-agnostic forced aligner that runs alongside
TTS generation and emits ``(word, start_ms, end_ms)`` triples per audio
chunk. Default-off; opt-in via ``--enable-word-timestamps`` server flag
plus ``word_timestamps=true`` per request.

Layout:
    types.py          - data classes shared by the sidecar and its client
    ctc_decode.py     - pure-function CTC + char-to-word post-processing
    sidecar_proc.py   - subprocess entry that owns the aligner model
    sidecar_client.py - in-process client managing the subprocess lifecycle

PR-1 wires the sidecar into ``serving_speech_stream.py`` directly.
PR-2 swaps the subprocess for a native ``LLM_GENERATION`` stage worker
without changing any of the public protocol or CLI flags.
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
