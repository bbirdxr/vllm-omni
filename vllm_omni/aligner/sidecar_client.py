# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-process client managing the aligner sidecar subprocess lifecycle.

The streaming handler treats this client as a black box: ``submit`` to
enqueue an alignment job, ``await aligned`` to retrieve the result, and
``cancel`` to drop pending jobs when a request disconnects. PR-2 swaps
the underlying transport from ``multiprocessing.Queue`` to the
``OmniConnector`` layer; the public method signatures stay the same.

Failure semantics (intentional and stable across PR-1/PR-2):

* ``aligned`` resolves to ``None`` when the chunk could not be aligned
  (timeout, sidecar crash, decode error). The streaming layer maps this
  to JSON ``timestamps: null`` and continues serving audio.
* ``aligned`` resolves to ``[]`` (empty list) when the aligner returned
  no tokens for the chunk (silence). The streaming layer keeps this as
  ``timestamps: []`` so clients can distinguish "no speech" from
  "alignment failed".
* ``submit`` raises :class:`AlignerError` on synchronous failures
  (sidecar already dead, after shutdown). The streaming layer catches
  these and falls back to the binary-frame path for the rest of the
  request.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import queue
import time
from typing import Any

from vllm_omni.aligner.sidecar_proc import (
    SidecarArgs,
    aligner_sidecar_main,
)
from vllm_omni.aligner.types import (
    SHUTDOWN_SIGNAL,
    AlignerError,
    AlignerErrorKind,
    AlignmentRequest,
    AlignmentResponse,
)

logger = logging.getLogger(__name__)


_PendingKey = tuple[str, int, int]  # (req_id, sentence_index, chunk_id)


class AlignerSidecarClient:
    """Spawns the aligner sidecar and multiplexes per-chunk responses.

    Parameters:
        args: Plain-data subset of engine args (``SidecarArgs``).
        ready_timeout: Seconds to wait for the sidecar to load its model
            and signal readiness. The sidecar can take 10-20s to load
            ``Qwen3-ForcedAligner-0.6B``; default 60s leaves headroom.
        in_q_maxsize: Bound on the input queue. ``submit`` raises rather
            than block when full; the streaming layer falls back to
            binary frames in that case. Default 1024 is comfortably
            larger than any single TTS request's chunk count.
        dispatch_poll_interval_s: How often the dispatch loop wakes up
            to check whether the sidecar is still alive. Lower = faster
            crash detection at the cost of more CPU wakeups; default 1s.
    """

    def __init__(
        self,
        args: SidecarArgs,
        *,
        ready_timeout: float = 60.0,
        in_q_maxsize: int = 1024,
        dispatch_poll_interval_s: float = 1.0,
    ) -> None:
        self._args = args
        self._dispatch_poll_interval_s = dispatch_poll_interval_s

        # Use spawn over fork: the parent process has already imported
        # FastAPI / uvicorn / vLLM, and forking that state into the
        # aligner process leaks GPU contexts and descriptors.
        ctx = mp.get_context("spawn")
        self._in_q: mp.Queue = ctx.Queue(maxsize=in_q_maxsize)
        self._out_q: mp.Queue = ctx.Queue()
        self._ready_q: mp.Queue = ctx.Queue()

        self._proc = ctx.Process(
            target=aligner_sidecar_main,
            name="vllm-omni-aligner-sidecar",
            args=(self._in_q, self._out_q, self._ready_q, args),
            daemon=True,
        )
        self._proc.start()
        self._wait_ready(ready_timeout)

        # Pending futures live in a dict keyed by (req_id, sent_idx, chunk_id).
        # The dispatch loop pops entries on response; cancel() pops entries
        # eagerly. Both routes resolve the future to None or list[dict].
        self._pending: dict[_PendingKey, asyncio.Future] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._dispatch_task: asyncio.Task | None = None
        self._shutdown_started = False

    # ----- lifecycle -----

    def _wait_ready(self, timeout: float) -> None:
        """Block until the sidecar puts a READY/ERROR token on ready_q."""
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._terminate_proc()
                raise AlignerError(
                    f"Aligner sidecar did not become ready within {timeout}s. "
                    f"Check --aligner-model path and GPU memory.",
                    AlignerErrorKind.UNKNOWN,
                )
            try:
                tag, payload = self._ready_q.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                if not self._proc.is_alive():
                    raise AlignerError(
                        f"Aligner sidecar exited during startup (exit code {self._proc.exitcode}).",
                        AlignerErrorKind.PROCESS_DEAD,
                    )
                continue
            break

        if tag == "READY":
            logger.info("Aligner sidecar ready (pid=%s)", self._proc.pid)
            return
        if tag == "ERROR":
            self._terminate_proc()
            raise AlignerError(
                f"Aligner sidecar failed to start: {payload}",
                AlignerErrorKind.UNKNOWN,
            )
        self._terminate_proc()
        raise AlignerError(
            f"Aligner sidecar emitted unknown ready token: {tag!r}",
            AlignerErrorKind.UNKNOWN,
        )

    async def start_dispatcher(self) -> None:
        """Bind the running event loop and start the multiplexer task.

        Must be called from inside an asyncio context after the API
        server has booted. Idempotent: a second call is a no-op.
        """
        if self._dispatch_task is not None:
            return
        self._loop = asyncio.get_running_loop()
        self._dispatch_task = self._loop.create_task(self._dispatch_loop(), name="aligner-sidecar-dispatcher")

    async def shutdown(self, timeout: float = 5.0) -> None:
        """Stop the dispatcher, ask the sidecar to exit, then join."""
        if self._shutdown_started:
            return
        self._shutdown_started = True

        # Tell the sidecar to leave its busy loop. Best-effort: if the
        # queue is full or the sidecar is already gone, we still try to
        # stop the dispatcher and join the process.
        try:
            self._in_q.put(SHUTDOWN_SIGNAL, timeout=1.0)
        except Exception:  # noqa: BLE001
            pass

        if self._dispatch_task is not None:
            self._dispatch_task.cancel()
            try:
                await self._dispatch_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

        self._fail_all_pending("aligner sidecar shutting down", AlignerErrorKind.SUBMIT_AFTER_SHUTDOWN)

        if self._proc.is_alive():
            self._proc.join(timeout=timeout)
        if self._proc.is_alive():
            logger.warning("Aligner sidecar did not exit cleanly within %.1fs; terminating.", timeout)
            self._terminate_proc()

    # ----- public API used by the streaming layer -----

    def submit(
        self,
        req_id: str,
        sentence_index: int,
        chunk_id: int,
        audio: bytes,
        text: str,
        sample_rate: int,
    ) -> None:
        """Enqueue one alignment job. Non-blocking; raises on synchronous failure.

        Raises:
            AlignerError: Sidecar process is not alive, the input queue
                is full, or ``shutdown()`` has been called.
        """
        if self._shutdown_started:
            raise AlignerError(
                "submit after shutdown",
                AlignerErrorKind.SUBMIT_AFTER_SHUTDOWN,
            )
        if not self._proc.is_alive():
            raise AlignerError(
                f"aligner sidecar is dead (exit code {self._proc.exitcode})",
                AlignerErrorKind.PROCESS_DEAD,
            )
        if self._loop is None:
            raise AlignerError(
                "submit before start_dispatcher; call client.start_dispatcher() first",
                AlignerErrorKind.UNKNOWN,
            )

        key: _PendingKey = (req_id, sentence_index, chunk_id)
        fut: asyncio.Future = self._loop.create_future()
        self._pending[key] = fut

        request = AlignmentRequest(
            req_id=req_id,
            sentence_index=sentence_index,
            chunk_id=chunk_id,
            audio=audio,
            text=text,
            sample_rate=sample_rate,
        )
        try:
            self._in_q.put_nowait(request)
        except queue.Full:
            self._pending.pop(key, None)
            raise AlignerError(
                "aligner input queue is full; consider lowering concurrency or increasing in_q_maxsize",
                AlignerErrorKind.UNKNOWN,
            ) from None

    async def aligned(
        self,
        req_id: str,
        sentence_index: int,
        chunk_id: int,
    ) -> list[dict] | None:
        """Await the alignment result for one previously-submitted chunk.

        Returns:
            List of timestamp dicts on success (possibly empty for silence),
            or ``None`` if the alignment failed for any reason. Failures
            are logged inside the dispatcher; callers do not need to
            distinguish between "timed out", "decoder error", and
            "sidecar crashed" — the contract guarantees audio still
            flows in all three cases.
        """
        key: _PendingKey = (req_id, sentence_index, chunk_id)
        fut = self._pending.get(key)
        if fut is None:
            # Either submit was never called for this key, or cancel()
            # already dropped it. Treat as "no timestamps".
            return None
        return await fut

    def cancel(self, req_id: str) -> None:
        """Drop all pending futures for a request (used on client disconnect).

        Resolves each affected future to ``None`` rather than cancelling
        it, so any code currently awaiting :meth:`aligned` cleanly
        receives "no timestamps" instead of a CancelledError.
        """
        if not self._pending:
            return
        keys_to_drop = [k for k in self._pending if k[0] == req_id]
        for key in keys_to_drop:
            fut = self._pending.pop(key, None)
            if fut is not None and not fut.done():
                fut.set_result(None)

    def health_check(self) -> bool:
        """True iff the sidecar process is still running."""
        return self._proc.is_alive()

    # ----- dispatcher internals -----

    async def _dispatch_loop(self) -> None:
        """Pull responses off ``out_q`` and resolve the matching futures.

        Wakes up at least once per ``dispatch_poll_interval_s`` to detect
        a dead sidecar (the ``out_q.get`` call would otherwise hang
        forever once the producer is gone).
        """
        try:
            while not self._shutdown_started:
                resp = await self._get_next_response()
                if resp is None:
                    continue  # poll tick, no message
                self._dispatch_response(resp)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.exception("Aligner dispatch loop crashed; failing pending alignments.")
            self._fail_all_pending(
                "aligner dispatcher crashed",
                AlignerErrorKind.UNKNOWN,
            )

    async def _get_next_response(self) -> AlignmentResponse | None:
        """Block on ``out_q`` for up to one poll interval; detect crashes on timeout."""
        try:
            resp = await asyncio.to_thread(self._out_q.get, True, self._dispatch_poll_interval_s)
        except queue.Empty:
            if not self._proc.is_alive():
                logger.error(
                    "Aligner sidecar died (exit code %s); failing %d pending alignments",
                    self._proc.exitcode,
                    len(self._pending),
                )
                self._fail_all_pending(
                    f"aligner sidecar died (exit code {self._proc.exitcode})",
                    AlignerErrorKind.PROCESS_DEAD,
                )
                # Stop the loop: future submits will raise PROCESS_DEAD.
                self._shutdown_started = True
                return None
            return None

        if not isinstance(resp, AlignmentResponse):
            logger.warning(
                "Aligner dispatch loop received unexpected payload type: %s",
                type(resp).__name__,
            )
            return None
        return resp

    def _dispatch_response(self, resp: AlignmentResponse) -> None:
        key: _PendingKey = (resp.req_id, resp.sentence_index, resp.chunk_id)
        fut = self._pending.pop(key, None)
        if fut is None or fut.done():
            # Caller cancelled this chunk before the sidecar replied.
            return
        if resp.ok:
            fut.set_result(resp.timestamps)
        else:
            logger.debug(
                "Alignment failed for req=%s chunk=%s: %s",
                resp.req_id,
                resp.chunk_id,
                resp.error,
            )
            fut.set_result(None)

    def _fail_all_pending(self, reason: str, kind: AlignerErrorKind) -> None:
        """Resolve every pending future to ``None`` (does not raise)."""
        if not self._pending:
            return
        logger.warning(
            "Failing %d pending aligner futures: %s (%s)",
            len(self._pending),
            reason,
            kind.value,
        )
        for key in list(self._pending):
            fut = self._pending.pop(key, None)
            if fut is not None and not fut.done():
                fut.set_result(None)

    def _terminate_proc(self) -> None:
        """Force-kill the sidecar; used only from error paths."""
        if not self._proc.is_alive():
            return
        try:
            self._proc.terminate()
            self._proc.join(timeout=2.0)
        except Exception:  # noqa: BLE001
            pass
        if self._proc.is_alive():
            try:
                self._proc.kill()
            except Exception:  # noqa: BLE001
                pass


def build_client_from_engine_args(engine_args: Any) -> AlignerSidecarClient:
    """Helper: project engine args -> SidecarArgs -> client.

    The streaming setup code calls this from
    :class:`OmniOpenAIServingSpeech.__init__` so the API server fails
    fast when ``--enable-word-timestamps`` is set but the sidecar can't
    start.
    """
    sidecar_args = SidecarArgs.from_engine_args(engine_args)
    return AlignerSidecarClient(sidecar_args)
