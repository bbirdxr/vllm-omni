# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Data types shared between the aligner sidecar and its API-server client.

The sidecar process intentionally avoids importing pydantic, FastAPI, or
the vLLM runtime so it can start fast and stay isolated. Cross-process
payloads are plain Python dataclasses serialized by the underlying
transport (``multiprocessing.Queue`` in PR-1, ``OmniConnector`` in PR-2;
both rely on pickle / msgpack so dataclasses suffice).

Only types in this module live on the cross-process boundary. Everything
that touches HTTP/WebSocket schemas (e.g. ``WordTimestamp``) lives in
``vllm_omni.entrypoints.openai.protocol.audio`` to keep the API layer
free of internal aligner concerns.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class AlignmentRequest:
    """One alignment job: (audio chunk, ground-truth text) -> timestamps.

    The audio payload is raw PCM bytes — caller is responsible for
    matching ``sample_rate`` to whatever the TTS pipeline produced. The
    sidecar resamples internally if its model expects a different rate.
    """

    req_id: str
    sentence_index: int
    chunk_id: int
    audio: bytes
    text: str
    sample_rate: int


@dataclass(frozen=True, slots=True)
class AlignmentResponse:
    """Reply for one ``AlignmentRequest``.

    Success: ``ok=True``; ``timestamps`` is a list of dicts each carrying
    the keys ``word``, ``start_ms``, ``end_ms``, and ``confidence``. The
    list may legitimately be empty (silence / no aligned tokens), which
    the streaming layer surfaces to clients as ``timestamps: []`` —
    distinct from ``timestamps: null`` (failure).

    Failure: ``ok=False``; ``error`` carries a short reason and
    ``timestamps`` is empty. The streaming layer maps this to JSON
    ``"timestamps": null`` so the client can distinguish "no speech" from
    "alignment failed".
    """

    req_id: str
    sentence_index: int
    chunk_id: int
    ok: bool
    timestamps: list[dict] = field(default_factory=list)
    error: str | None = None


class AlignerErrorKind(str, enum.Enum):
    """Stable identifiers for client-side metric / log keying.

    Kept as a ``str`` enum so the value can be used directly as a
    Prometheus label or log field without a separate ``.value`` lookup.
    """

    PROCESS_DEAD = "process_dead"
    SUBMIT_AFTER_SHUTDOWN = "submit_after_shutdown"
    TIMEOUT = "timeout"
    DECODE_FAILED = "decode_failed"
    UNKNOWN = "unknown"


class AlignerError(RuntimeError):
    """Raised by ``AlignerClient`` when the sidecar cannot serve a request.

    Callers should treat this as *"no timestamps for this chunk; audio
    still flows"* rather than *"the speech request failed"*. The
    streaming layer catches this and emits ``timestamps: null`` to the
    WebSocket client.
    """

    def __init__(
        self,
        message: str,
        kind: AlignerErrorKind = AlignerErrorKind.UNKNOWN,
    ) -> None:
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True, slots=True)
class _ShutdownSignal:
    """Sentinel placed on the input queue to ask the sidecar to exit cleanly."""


SHUTDOWN_SIGNAL = _ShutdownSignal()
