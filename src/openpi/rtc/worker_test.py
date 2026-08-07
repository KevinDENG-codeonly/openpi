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
