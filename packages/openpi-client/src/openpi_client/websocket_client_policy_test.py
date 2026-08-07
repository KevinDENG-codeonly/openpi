"""Tests for cancellable websocket policy transport setup and shutdown."""

import threading

from openpi_client import websocket_client_policy


def test_close_is_idempotent_and_closes_the_active_connection(monkeypatch):
    class FakeConnection:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    connection = FakeConnection()

    def wait_for_server(self):
        self._ws = connection  # noqa: SLF001
        return connection, {"name": "fake"}

    monkeypatch.setattr(websocket_client_policy.WebsocketClientPolicy, "_wait_for_server", wait_for_server)
    policy = websocket_client_policy.WebsocketClientPolicy()

    policy.close()
    policy.close()

    assert connection.close_calls == 1


def test_connect_attempt_passes_an_explicit_open_timeout(monkeypatch):
    connect_calls = []

    class FakeConnection:
        def recv(self, *, timeout):
            return b"metadata"

        def close(self):
            pass

    def connect(*args, **kwargs):
        connect_calls.append((args, kwargs))
        return FakeConnection()

    monkeypatch.setattr(websocket_client_policy.websockets.sync.client, "connect", connect)
    monkeypatch.setattr(websocket_client_policy.msgpack_numpy, "unpackb", lambda _value: {"name": "fake"})

    policy = websocket_client_policy.WebsocketClientPolicy(connect_timeout_s=0.25)
    policy.close()

    assert connect_calls[0][1]["open_timeout"] == 0.25


def test_server_wait_stops_immediately_when_cancelled_during_retry(monkeypatch):
    cancel_event = threading.Event()
    attempted_connection = threading.Event()
    errors = []

    def refuse_connection(*args, **kwargs):
        attempted_connection.set()
        raise ConnectionRefusedError()

    monkeypatch.setattr(websocket_client_policy.websockets.sync.client, "connect", refuse_connection)

    def construct_policy():
        try:
            websocket_client_policy.WebsocketClientPolicy(cancel_event=cancel_event)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=construct_policy)
    thread.start()
    assert attempted_connection.wait(timeout=1)

    cancel_event.set()
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], websocket_client_policy.PolicyCancelledError)


def test_server_wait_closes_a_connection_created_after_cancellation(monkeypatch):
    cancel_event = threading.Event()
    connection_attempt_started = threading.Event()
    errors = []

    class FakeConnection:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    connection = FakeConnection()

    def connect_after_cancellation(*args, **kwargs):
        connection_attempt_started.set()
        assert cancel_event.wait(timeout=1)
        return connection

    monkeypatch.setattr(websocket_client_policy.websockets.sync.client, "connect", connect_after_cancellation)

    def construct_policy():
        try:
            websocket_client_policy.WebsocketClientPolicy(cancel_event=cancel_event)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=construct_policy)
    thread.start()
    assert connection_attempt_started.wait(timeout=1)

    cancel_event.set()
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert connection.close_calls == 1
    assert len(errors) == 1
    assert isinstance(errors[0], websocket_client_policy.PolicyCancelledError)


def test_server_wait_closes_metadata_receive_when_cancelled(monkeypatch):
    cancel_event = threading.Event()
    metadata_receive_started = threading.Event()
    errors = []

    class FakeConnection:
        def __init__(self):
            self.close_calls = 0

        def recv(self, *, timeout=None):
            metadata_receive_started.set()
            if timeout is None:
                raise AssertionError("metadata receive must be polled with a cancellation-aware timeout")
            assert cancel_event.wait(timeout=1)
            raise TimeoutError()

        def close(self):
            self.close_calls += 1

    connection = FakeConnection()
    monkeypatch.setattr(
        websocket_client_policy.websockets.sync.client,
        "connect",
        lambda *args, **kwargs: connection,
    )

    def construct_policy():
        try:
            websocket_client_policy.WebsocketClientPolicy(cancel_event=cancel_event)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=construct_policy)
    thread.start()
    assert metadata_receive_started.wait(timeout=1)

    cancel_event.set()
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert connection.close_calls == 1
    assert len(errors) == 1
    assert isinstance(errors[0], websocket_client_policy.PolicyCancelledError)
