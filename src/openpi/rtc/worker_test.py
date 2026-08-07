"""Tests for the single-flight RTC inference worker."""

import threading
import time

import pytest

from openpi.rtc.worker import RTCInferenceResult
from openpi.rtc.worker import RTCInferenceWorker
from openpi.rtc.worker import RTCWorkerBusyError


def test_submit_returns_value_with_monotonic_timestamps():
    worker = RTCInferenceWorker(lambda request: request.upper())

    try:
        result = worker.submit("fold").result(timeout=1)
    finally:
        worker.close()

    assert isinstance(result, RTCInferenceResult)
    assert result.value == "FOLD"
    assert result.started_at <= result.finished_at
    assert result.finished_at <= time.monotonic()


def test_submit_rejects_second_request_while_first_is_in_flight():
    started = threading.Event()
    allow_finish = threading.Event()

    def infer(request: str) -> str:
        started.set()
        assert allow_finish.wait(timeout=1)
        return request.upper()

    worker = RTCInferenceWorker(infer)
    try:
        first = worker.submit("first")
        assert started.wait(timeout=1)

        with pytest.raises(RTCWorkerBusyError, match="in flight"):
            worker.submit("second")

        allow_finish.set()
        assert first.result(timeout=1).value == "FIRST"
    finally:
        allow_finish.set()
        worker.close()


def test_submit_allows_a_new_request_after_completion():
    worker = RTCInferenceWorker(lambda request: request.upper())

    try:
        assert worker.submit("first").result(timeout=1).value == "FIRST"
        assert worker.submit("second").result(timeout=1).value == "SECOND"
    finally:
        worker.close()


def test_inference_exception_propagates_and_allows_later_recovery():
    def infer(request: str) -> str:
        if request == "bad":
            raise ValueError("invalid request")
        return request.upper()

    worker = RTCInferenceWorker(infer)

    try:
        with pytest.raises(ValueError, match="invalid request"):
            worker.submit("bad").result(timeout=1)

        assert worker.submit("recovered").result(timeout=1).value == "RECOVERED"
    finally:
        worker.close()


def test_close_waits_rejects_later_submits_and_is_idempotent():
    started = threading.Event()
    allow_finish = threading.Event()
    shutdown_started = threading.Event()
    close_returned = threading.Event()

    def infer(request: str) -> str:
        started.set()
        assert allow_finish.wait(timeout=1)
        return request.upper()

    worker = RTCInferenceWorker(infer)
    original_shutdown = worker._executor.shutdown  # noqa: SLF001

    def record_shutdown(*, wait: bool, cancel_futures: bool) -> None:
        assert wait is True
        assert cancel_futures is False
        shutdown_started.set()
        original_shutdown(wait=wait, cancel_futures=cancel_futures)

    worker._executor.shutdown = record_shutdown  # noqa: SLF001
    future = worker.submit("close")
    assert started.wait(timeout=1)

    close_thread = threading.Thread(target=lambda: (worker.close(), close_returned.set()))
    close_thread.start()
    assert shutdown_started.wait(timeout=1)
    assert not close_returned.is_set()

    allow_finish.set()
    close_thread.join(timeout=1)

    assert not close_thread.is_alive()
    assert close_returned.is_set()
    assert future.result(timeout=1).value == "CLOSE"
    worker.close()
    with pytest.raises(RuntimeError, match="closed"):
        worker.submit("later")


def test_nonblocking_close_releases_the_caller_while_inference_finishes_later():
    started = threading.Event()
    allow_finish = threading.Event()

    def infer(request: str) -> str:
        started.set()
        assert allow_finish.wait(timeout=1)
        return request.upper()

    worker = RTCInferenceWorker(infer)
    future = worker.submit("slow")
    assert started.wait(timeout=1)

    try:
        worker.close(wait=False)

        assert not future.done()
        with pytest.raises(RuntimeError, match="closed"):
            worker.submit("later")
    finally:
        allow_finish.set()
        worker.close()

    assert future.result(timeout=1).value == "SLOW"


def test_close_from_its_own_inference_thread_is_rejected_without_closing():
    shutdown_callers: list[threading.Thread] = []

    def infer(request: str) -> str:
        with pytest.raises(RuntimeError, match="own inference thread"):
            worker.close()
        return request.upper()

    worker = RTCInferenceWorker(infer)
    original_shutdown = worker._executor.shutdown  # noqa: SLF001

    def record_shutdown(*, wait: bool, cancel_futures: bool) -> None:
        assert wait is True
        assert cancel_futures is False
        shutdown_callers.append(threading.current_thread())
        original_shutdown(wait=wait, cancel_futures=cancel_futures)

    worker._executor.shutdown = record_shutdown  # noqa: SLF001

    assert worker.submit("self-close").result(timeout=1).value == "SELF-CLOSE"
    assert shutdown_callers == []

    worker.close()

    assert shutdown_callers == [threading.current_thread()]


def test_concurrent_external_closes_wait_and_block_new_submissions():
    started = threading.Event()
    allow_finish = threading.Event()
    shutdown_started = threading.Event()
    second_close_waiting = threading.Event()
    first_close_returned = threading.Event()
    second_close_returned = threading.Event()

    def infer(request: str) -> str:
        started.set()
        assert allow_finish.wait(timeout=1)
        return request.upper()

    worker = RTCInferenceWorker(infer)
    original_shutdown = worker._executor.shutdown  # noqa: SLF001
    original_wait = worker._close_complete.wait  # noqa: SLF001

    def record_shutdown(*, wait: bool, cancel_futures: bool) -> None:
        assert wait is True
        assert cancel_futures is False
        shutdown_started.set()
        original_shutdown(wait=wait, cancel_futures=cancel_futures)

    def record_wait(timeout: float | None = None) -> bool:
        second_close_waiting.set()
        return original_wait(timeout)

    worker._executor.shutdown = record_shutdown  # noqa: SLF001
    worker._close_complete.wait = record_wait  # noqa: SLF001
    future = worker.submit("close")
    assert started.wait(timeout=1)

    first_close = threading.Thread(target=lambda: (worker.close(), first_close_returned.set()))
    second_close = threading.Thread(target=lambda: (worker.close(), second_close_returned.set()))
    try:
        first_close.start()
        assert shutdown_started.wait(timeout=1)
        with pytest.raises(RuntimeError, match="closed"):
            worker.submit("later")

        second_close.start()
        assert second_close_waiting.wait(timeout=1)
        assert not first_close_returned.is_set()
        assert not second_close_returned.is_set()

        allow_finish.set()
        first_close.join(timeout=1)
        second_close.join(timeout=1)

        assert not first_close.is_alive()
        assert not second_close.is_alive()
        assert first_close_returned.is_set()
        assert second_close_returned.is_set()
        assert future.result(timeout=1).value == "CLOSE"
    finally:
        allow_finish.set()
        first_close.join(timeout=1)
        second_close.join(timeout=1)
