"""Generic single-flight worker for RTC inference."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
import dataclasses
import threading
import time
from typing import Generic, TypeVar

RequestT = TypeVar("RequestT")
ValueT = TypeVar("ValueT")


@dataclasses.dataclass(frozen=True)
class RTCInferenceResult(Generic[ValueT]):
    """The value returned by inference and its monotonic execution interval."""

    value: ValueT
    started_at: float
    finished_at: float


class RTCWorkerBusyError(RuntimeError):
    """Raised when inference is already in flight."""


class RTCInferenceWorker(Generic[RequestT, ValueT]):
    """Run inference in a dedicated worker thread."""

    def __init__(self, infer: Callable[[RequestT], ValueT]) -> None:
        self._infer = infer
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future: Future[RTCInferenceResult[ValueT]] | None = None
        self._closed = False
        self._nonblocking_shutdown = False
        self._close_complete = threading.Event()
        self._lock = threading.Lock()
        self._inference_thread: threading.Thread | None = None

    def submit(self, request: RequestT) -> Future[RTCInferenceResult[ValueT]]:
        """Start inference unless a request is already in flight."""
        with self._lock:
            if self._closed:
                raise RuntimeError("RTC inference worker is closed")
            if self._future is not None and not self._future.done():
                raise RTCWorkerBusyError("RTC inference worker already has a request in flight")
            self._future = self._executor.submit(self._run, request)
            return self._future

    def close(self, *, wait: bool = True) -> None:
        """Close the worker, optionally returning while one inference finishes."""
        if threading.current_thread() is self._inference_thread:
            raise RuntimeError("RTC inference worker cannot be closed from its own inference thread")
        with self._lock:
            if self._closed:
                should_shutdown = False
                should_complete_nonblocking_shutdown = wait and self._nonblocking_shutdown
                if should_complete_nonblocking_shutdown:
                    self._nonblocking_shutdown = False
            else:
                self._closed = True
                should_shutdown = True
                should_complete_nonblocking_shutdown = False
                self._nonblocking_shutdown = not wait
        if should_shutdown:
            try:
                self._executor.shutdown(wait=wait, cancel_futures=False)
            finally:
                if wait:
                    self._close_complete.set()
                else:
                    future = self._future
                    if future is None:
                        self._close_complete.set()
                    else:
                        future.add_done_callback(lambda _future: self._close_complete.set())
        elif should_complete_nonblocking_shutdown:
            try:
                self._executor.shutdown(wait=True, cancel_futures=False)
            finally:
                self._close_complete.set()
        elif wait:
            self._close_complete.wait()

    def _run(self, request: RequestT) -> RTCInferenceResult[ValueT]:
        self._inference_thread = threading.current_thread()
        started_at = time.monotonic()
        try:
            value = self._infer(request)
        finally:
            finished_at = time.monotonic()
        return RTCInferenceResult(value=value, started_at=started_at, finished_at=finished_at)
