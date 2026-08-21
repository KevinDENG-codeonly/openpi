"""Tests for cancellable websocket policy transport setup and shutdown."""

import threading
import socket
from types import SimpleNamespace

import pytest

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

    class FakeRawSocket:
        def send(self, data, flags):
            raise AssertionError("the connection setup must not send data")

    class FakeConnection:
        def __init__(self):
            self.socket = FakeRawSocket()

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
    connection = policy._ws  # noqa: SLF001
    policy.close()

    assert connect_calls[0][1]["open_timeout"] == 0.25
    assert isinstance(connection.socket, websocket_client_policy._SocketWriteDeadlineProxy)  # noqa: SLF001


def test_explicit_connect_timeout_fails_when_write_timeout_cannot_be_configured(monkeypatch):
    class UnusableSocket:
        pass

    class FakeConnection:
        def __init__(self):
            self.socket = UnusableSocket()
            self.close_calls = 0

        def recv(self, *, timeout):
            return b"metadata"

        def close(self):
            self.close_calls += 1

    connection = FakeConnection()
    monkeypatch.setattr(
        websocket_client_policy.websockets.sync.client,
        "connect",
        lambda *args, **kwargs: connection,
    )

    with pytest.raises(RuntimeError, match="total write deadline"):
        websocket_client_policy.WebsocketClientPolicy(connect_timeout_s=0.25)

    assert connection.close_calls == 1


def test_finite_connect_timeout_rejects_tls_socket_before_any_inference_write(monkeypatch):
    class FakeTLSSocket:
        def send(self, data, flags):
            raise AssertionError("TLS socket must not be wrapped for MSG_DONTWAIT sends")

    class FakeConnection:
        def __init__(self):
            self.socket = FakeTLSSocket()
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    connection = FakeConnection()
    monkeypatch.setattr(
        websocket_client_policy,
        "ssl",
        SimpleNamespace(SSLSocket=FakeTLSSocket),
        raising=False,
    )
    monkeypatch.setattr(
        websocket_client_policy.websockets.sync.client,
        "connect",
        lambda *args, **kwargs: connection,
    )

    with pytest.raises(RuntimeError, match="non-TLS ws://"):
        websocket_client_policy.WebsocketClientPolicy(host="wss://policy", connect_timeout_s=0.25)

    assert connection.close_calls == 1


def test_wss_without_finite_connect_timeout_preserves_existing_policy_behavior(monkeypatch):
    class FakeTLSSocket:
        pass

    class FakeConnection:
        def __init__(self):
            self.socket = FakeTLSSocket()

        def recv(self, *, timeout):
            return b"metadata"

        def close(self):
            pass

    connection = FakeConnection()
    monkeypatch.setattr(
        websocket_client_policy,
        "ssl",
        SimpleNamespace(SSLSocket=FakeTLSSocket),
        raising=False,
    )
    monkeypatch.setattr(
        websocket_client_policy.websockets.sync.client,
        "connect",
        lambda *args, **kwargs: connection,
    )
    monkeypatch.setattr(websocket_client_policy.msgpack_numpy, "unpackb", lambda _value: {"name": "fake"})

    policy = websocket_client_policy.WebsocketClientPolicy(host="wss://policy")
    policy.close()

    assert isinstance(connection.socket, FakeTLSSocket)


def test_policy_write_deadline_proxy_does_not_reset_its_deadline_after_partial_writes(monkeypatch):
    clock = [0.0]
    select_timeouts = []

    class FakeRawSocket:
        def __init__(self):
            self.send_calls = []
            self.shutdown_calls = []
            self.close_calls = 0

        def send(self, data, flags):
            self.send_calls.append((bytes(data), flags))
            clock[0] += 0.25
            return 1

        def recv(self, size):
            return b"received"

        def fileno(self):
            return 42

        def shutdown(self, how):
            self.shutdown_calls.append(how)
            return "shutdown"

        def close(self):
            self.close_calls += 1

    raw_socket = FakeRawSocket()

    def monotonic():
        return clock[0]

    def select(readable, writable, exceptional, timeout):
        assert readable == []
        assert writable == [raw_socket]
        assert exceptional == []
        select_timeouts.append(timeout)
        if len(select_timeouts) < 3:
            return [], [raw_socket], []
        clock[0] += timeout
        return [], [], []

    monkeypatch.setattr(websocket_client_policy.time, "monotonic", monotonic)
    monkeypatch.setattr(websocket_client_policy.select, "select", select)
    proxy = websocket_client_policy._SocketWriteDeadlineProxy(raw_socket, timeout_s=1.0)  # noqa: SLF001

    with pytest.raises(TimeoutError, match="Websocket send timed out"):
        proxy.sendall(b"abcd", flags=0x100)

    assert select_timeouts == pytest.approx([1.0, 0.75, 0.5])
    assert raw_socket.send_calls == [
        (b"abcd", 0x100 | socket.MSG_DONTWAIT),
        (b"bcd", 0x100 | socket.MSG_DONTWAIT),
    ]
    assert proxy.recv(1) == b"received"
    assert proxy.fileno() == 42
    assert proxy.shutdown(socket.SHUT_RDWR) == "shutdown"
    proxy.close()
    assert raw_socket.shutdown_calls == [socket.SHUT_RDWR]
    assert raw_socket.close_calls == 1


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
