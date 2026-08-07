import logging
import threading
from typing import Dict, Optional, Tuple

from typing_extensions import override
import websockets.sync.client

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy

_SERVER_METADATA_POLL_TIMEOUT_S = 0.1


class PolicyCancelledError(RuntimeError):
    """Raised when a policy connection or inference is cancelled during shutdown."""


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._cancel_event = cancel_event or threading.Event()
        self._state_lock = threading.RLock()
        self._closed = False
        self._ws: Optional[websockets.sync.client.ClientConnection] = None
        self._server_metadata: Dict = {}
        _connection, self._server_metadata = self._wait_for_server()
        self._raise_if_cancelled()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            self._raise_if_cancelled()
            try:
                # A synchronous connect call is not force-interruptible; once it returns,
                # cancellation is checked and the raced connection is closed immediately.
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers
                )
                self._register_connection(conn)
                metadata = self._recv_server_metadata(conn)
                self._raise_if_cancelled()
                return conn, metadata
            except ConnectionRefusedError:
                self._raise_if_cancelled()
                logging.info("Still waiting for server...")
                if self._cancel_event.wait(timeout=5):
                    self._raise_if_cancelled()

    def _recv_server_metadata(self, connection: websockets.sync.client.ClientConnection) -> Dict:
        while True:
            try:
                return msgpack_numpy.unpackb(connection.recv(timeout=_SERVER_METADATA_POLL_TIMEOUT_S))
            except TimeoutError:
                self._raise_if_cancelled()
            except Exception:
                if self._is_cancelled():
                    self._raise_if_cancelled()
                raise

    def close(self) -> None:
        """Idempotently cancel and close the active websocket transport."""
        self._cancel_event.set()
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            connection = self._ws
            self._ws = None
        if connection is not None:
            try:
                connection.close()
            except Exception:  # noqa: BLE001
                logging.debug("Ignoring websocket close failure during policy shutdown.", exc_info=True)

    def _is_cancelled(self) -> bool:
        with self._state_lock:
            return self._closed or self._cancel_event.is_set()

    def _raise_if_cancelled(self) -> None:
        if self._is_cancelled():
            self.close()
            raise PolicyCancelledError("Policy connection was cancelled.")

    def _register_connection(self, connection: websockets.sync.client.ClientConnection) -> None:
        with self._state_lock:
            cancelled = self._closed or self._cancel_event.is_set()
            if not cancelled:
                self._ws = connection
        if cancelled:
            try:
                connection.close()
            except Exception:  # noqa: BLE001
                logging.debug("Ignoring websocket close failure during policy cancellation.", exc_info=True)
            raise PolicyCancelledError("Policy connection was cancelled while the websocket was being created.")

    @override
    def infer(
        self, obs: Dict, *, rtc: Optional[Dict] = None, return_model_actions: bool = False
    ) -> Dict:  # noqa: UP006
        self._raise_if_cancelled()
        if rtc is not None or return_model_actions:
            data = self._packer.pack(
                {
                    "obs": obs,
                    "rtc": rtc,
                    "return_model_actions": return_model_actions,
                }
            )
        else:
            data = self._packer.pack(obs)
        with self._state_lock:
            connection = self._ws
        if connection is None:
            raise PolicyCancelledError("Policy connection is closed.")
        try:
            connection.send(data)
            response = connection.recv()
        except Exception as exc:
            if self._is_cancelled():
                raise PolicyCancelledError("Policy inference was cancelled while waiting for a response.") from exc
            raise
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    @override
    def reset(self) -> None:
        pass
