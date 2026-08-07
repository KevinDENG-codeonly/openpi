"""Run the SpiritAI robot with asynchronous training-time RTC."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
import contextlib
import dataclasses
import errno
import logging
import math
import numbers
from pathlib import Path
import select
import socket
import threading
import time
from typing import Literal

import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro
import websockets.sync.client

from openpi.policies import spiritai_bridge
from openpi.rtc import ActionPlan
from openpi.rtc import DispatchAction
from openpi.rtc import RTCController
from openpi.rtc import RTCRequest
from openpi.rtc.runtime_config import RuntimeConfig
from openpi.rtc.runtime_config import load_runtime_config
from openpi.rtc.worker import RTCInferenceResult
from openpi.rtc.worker import RTCInferenceWorker

DEFAULT_RUNTIME_CONFIG = Path(__file__).parent / "configs" / "rtc" / "training_time.yaml"


@dataclasses.dataclass(frozen=True)
class BootstrapArgs:
    config: Path = DEFAULT_RUNTIME_CONFIG
    dry_run: bool = False


@dataclasses.dataclass(frozen=True)
class PolicyChunk:
    """Result of a single policy inference call."""

    actions: np.ndarray
    model_actions: np.ndarray | None = None


@dataclasses.dataclass(frozen=True)
class RTCRuntimeMetadata:
    """Validated model-space limits advertised by the policy server."""

    action_horizon: int
    action_dim: int
    training_max_delay_steps: int


@dataclasses.dataclass(frozen=True)
class RTCInferenceTask:
    """Worker-owned inference input frozen to an RTC request and robot snapshot."""

    request: RTCRequest
    obs: Mapping[str, object]
    images: Mapping[str, object]


@dataclasses.dataclass(frozen=True)
class PolicyBootstrapTask:
    """Ask the policy worker to connect and validate its RTC metadata."""


@dataclasses.dataclass(frozen=True)
class InitialInferenceTask:
    """Ask the policy worker for the first action plan after robot preflight."""

    obs: Mapping[str, object]
    images: Mapping[str, object]


PolicyWorkerTask = PolicyBootstrapTask | InitialInferenceTask | RTCInferenceTask
PolicyWorkerResult = RTCRuntimeMetadata | ActionPlan


@dataclasses.dataclass
class RTCInferenceFlight:
    """The only worker future allowed to use the policy connection at a time."""

    request: RTCRequest
    future: Future[RTCInferenceResult[ActionPlan]]
    expired: bool = False

    def expire_if_due(self, controller: RTCController, *, current_tick: int) -> bool:
        """Release controller state once a result was not observed by its deadline."""
        if self.expired:
            return False
        if controller.expire_request(self.request, current_tick=current_tick):
            self.expired = True
            return True
        return False

    def completed_result(self) -> RTCInferenceResult[ActionPlan] | None:
        """Consume a completed future, discarding its result when already expired."""
        result = self.future.result()
        return None if self.expired else result


def _positive_capability_integer(capability: Mapping[str, object], name: str) -> int:
    value = capability.get(name)
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0:
        raise ValueError(f"RTC capability {name} must be a positive integer, got {value!r}")
    return int(value)


def _nonnegative_capability_integer(capability: Mapping[str, object], name: str) -> int:
    value = capability.get(name)
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
        raise ValueError(f"RTC capability {name} must be a nonnegative integer, got {value!r}")
    return int(value)


def validate_rtc_runtime_metadata(runtime: RuntimeConfig, metadata: Mapping[str, object]) -> RTCRuntimeMetadata:
    """Require training-time RTC capabilities before opening the robot connection."""
    if not isinstance(metadata, Mapping):
        raise ValueError("Policy metadata must be a mapping with rtc_capabilities.")
    capability = metadata.get("rtc_capabilities")
    if not isinstance(capability, Mapping):
        raise ValueError("Policy metadata is missing rtc_capabilities.")
    if capability.get("algorithm") != "training_time_v1":
        raise ValueError("RTC capability algorithm must be training_time_v1.")

    action_horizon = _positive_capability_integer(capability, "action_horizon")
    action_dim = _positive_capability_integer(capability, "action_dim")
    training_max_delay_steps = _nonnegative_capability_integer(capability, "training_max_delay_steps")
    planned_max_steps = runtime.rtc.delay.planned_max_steps
    if (
        isinstance(planned_max_steps, bool)
        or not isinstance(planned_max_steps, numbers.Integral)
        or planned_max_steps <= 0
    ):
        raise ValueError(f"runtime.rtc.delay.planned_max_steps must be a positive integer, got {planned_max_steps!r}")
    if planned_max_steps > training_max_delay_steps:
        raise ValueError(
            "runtime.rtc.delay.planned_max_steps exceeds policy training_max_delay_steps: "
            f"{planned_max_steps} > {training_max_delay_steps}"
        )
    return RTCRuntimeMetadata(
        action_horizon=action_horizon,
        action_dim=action_dim,
        training_max_delay_steps=training_max_delay_steps,
    )


def _open_robot_connection(
    robot_url: str, *, timeout_s: float
) -> websockets.sync.client.ClientConnection:
    """Open and configure the robot transport before its first send."""
    connection = websockets.sync.client.connect(
        robot_url,
        max_size=None,
        compression=None,
        open_timeout=timeout_s,
    )
    try:
        _install_total_write_deadline(connection, timeout_s=timeout_s)
    except Exception:
        with contextlib.suppress(Exception):
            connection.close()
        raise
    return connection


class _SocketWriteDeadlineProxy:
    """Delegate a socket while enforcing one Linux send deadline per ``sendall`` call.

    ``MSG_DONTWAIT`` is a Linux per-send flag, unlike changing ``O_NONBLOCK`` or
    calling ``socket.settimeout()`` on the shared descriptor. Consequently, the
    websocket receiver thread continues to use the raw socket's normal receive
    behavior while each write is bounded by one monotonic deadline.
    """

    def __init__(self, raw_socket: socket.socket, *, timeout_s: float) -> None:
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, numbers.Real) or not math.isfinite(timeout_s):
            raise ValueError("websocket write deadline must be a positive finite number")
        if timeout_s <= 0:
            raise ValueError("websocket write deadline must be positive")
        if not hasattr(socket, "MSG_DONTWAIT"):
            raise RuntimeError("Websocket total write deadline requires Linux MSG_DONTWAIT.")
        self._raw_socket = raw_socket
        self._timeout_s = float(timeout_s)

    def __getattr__(self, name: str) -> object:
        return getattr(self._raw_socket, name)

    def sendall(self, data: bytes, flags: int = 0) -> None:
        payload = memoryview(data)
        deadline = time.monotonic() + self._timeout_s
        while payload:
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                raise TimeoutError("Websocket send timed out before the full buffer was written.")
            _readable, writable, _exceptional = select.select([], [self._raw_socket], [], remaining_s)
            if not writable:
                raise TimeoutError("Websocket send timed out before the full buffer was written.")
            try:
                sent = self._raw_socket.send(payload, flags | socket.MSG_DONTWAIT)
            except BlockingIOError:
                continue
            except OSError as exc:
                if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                    continue
                raise
            if sent <= 0:
                raise OSError("Websocket socket made a zero-byte write.")
            payload = payload[sent:]


def _install_total_write_deadline(
    connection: websockets.sync.client.ClientConnection, *, timeout_s: float
) -> None:
    """Install a Linux total-deadline send proxy before the first application write."""
    raw_socket = getattr(connection, "socket", None)
    if not callable(getattr(raw_socket, "send", None)):
        raise RuntimeError("Websocket connection does not expose a usable socket for total write deadline.")
    connection.socket = _SocketWriteDeadlineProxy(raw_socket, timeout_s=timeout_s)


def _recv_robot_response(
    robot_ws: websockets.sync.client.ClientConnection,
    *,
    timeout_s: float,
    operation: str,
) -> dict:
    """Receive one robot response within the configured safety bound."""
    try:
        return spiritai_bridge.unpack_robot_server_message(robot_ws.recv(timeout=timeout_s))
    except TimeoutError as exc:
        raise RuntimeError(f"{operation} timed out after {timeout_s:g}s") from exc


def _wait_until_robot_idle(
    robot_ws: websockets.sync.client.ClientConnection,
    busy_sleep_s: float,
    *,
    timeout_s: float,
    idle_timeout_s: float,
) -> None:
    deadline = time.monotonic() + idle_timeout_s
    while True:
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            raise RuntimeError(f"robot did not become idle within {idle_timeout_s:g}s")
        robot_ws.send(spiritai_bridge.pack_robot_server_message({"type": "get_status"}))
        status = _recv_robot_response(
            robot_ws,
            timeout_s=min(timeout_s, remaining_s),
            operation="robot status response",
        )
        if not status["busy"]:
            return
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            raise RuntimeError(f"robot did not become idle within {idle_timeout_s:g}s")
        time.sleep(min(busy_sleep_s, remaining_s))


def _get_robot_obs(
    robot_ws: websockets.sync.client.ClientConnection, *, timeout_s: float
) -> tuple[dict, dict]:
    robot_ws.send(spiritai_bridge.pack_robot_server_message({"type": "get_obs"}))
    msg = _recv_robot_response(
        robot_ws,
        timeout_s=timeout_s,
        operation="robot observation response",
    )
    if msg.get("type") not in (None, "obs"):
        raise spiritai_bridge.RobotServerProtocolError(f"Unexpected get_obs response: {msg.get('type')}")
    return msg["obs"], msg["images"]


def _wait_for_robot_command_ack(
    robot_ws: websockets.sync.client.ClientConnection,
    *,
    timeout_s: float,
    operation: str,
) -> dict:
    """Receive one command acknowledgement within the configured safety bound."""
    return _recv_robot_response(robot_ws, timeout_s=timeout_s, operation=f"{operation} ACK")


def _set_robot_external_following(
    robot_ws: websockets.sync.client.ClientConnection, *, enabled: bool, timeout_s: float
) -> None:
    robot_ws.send(spiritai_bridge.pack_robot_server_message({"type": "set_external_following", "enabled": enabled}))
    msg = _recv_robot_response(
        robot_ws,
        timeout_s=timeout_s,
        operation="external-following response",
    )
    if msg.get("type") == "error":
        raise spiritai_bridge.RobotServerProtocolError(
            f"robot_server rejected set_external_following: {msg.get('code')} {msg.get('message')}"
        )
    if msg.get("type") != "external_following":
        raise spiritai_bridge.RobotServerProtocolError(f"Unexpected set_external_following response: {msg.get('type')}")
    if not msg.get("accepted", False):
        raise spiritai_bridge.RobotServerProtocolError(
            f"robot_server did not enable external following: enabled={msg.get('enabled')} error={msg.get('error')}"
        )
    logging.info("Robot external following enabled: %s", msg.get("enabled"))


def _get_scalar_obs_value(obs: Mapping[str, object], key: str) -> float:
    value = np.asarray(obs[key], dtype=np.float32).reshape(-1)
    if value.size != 1:
        raise spiritai_bridge.RobotServerProtocolError(f"Expected scalar obs value for {key}, got shape {value.shape}")
    return float(value[0])


def _get_gripper_state(obs: Mapping[str, object]) -> tuple[float, float]:
    return (
        _get_scalar_obs_value(obs, "leftarm_gripper_state_pos"),
        _get_scalar_obs_value(obs, "rightarm_gripper_state_pos"),
    )


def _grippers_at_initial_state(
    obs: Mapping[str, object],
    *,
    initial_gripper_obs_state: float,
    tolerance: float,
) -> bool:
    left_gripper, right_gripper = _get_gripper_state(obs)
    return (
        abs(left_gripper - initial_gripper_obs_state) <= tolerance
        and abs(right_gripper - initial_gripper_obs_state) <= tolerance
    )


def _send_initial_gripper_reset(
    robot_ws: websockets.sync.client.ClientConnection,
    obs: Mapping[str, object],
    *,
    policy_action_layout: Literal["joint", "cartesian"],
    robot_command_kind: str,
    command_dim: int,
    gripper_reset_command_state: float,
    gripper_reset_steps: int,
    source_hz: float,
    command_ack_timeout_s: float,
    dry_run: bool,
) -> bool:
    if policy_action_layout == "cartesian":
        reset_command = spiritai_bridge.robot_server_obs_to_cartesian_command_layout(obs, command_dim)
        command_slices = spiritai_bridge.cartesian_command_slices(command_dim)
    else:
        reset_command = spiritai_bridge.robot_server_obs_to_joint_command_layout(obs, command_dim)
        command_slices = spiritai_bridge.joint_command_slices(command_dim)

    reset_command = reset_command.copy()
    reset_command[command_slices["left_gripper"]] = gripper_reset_command_state
    reset_command[command_slices["right_gripper"]] = gripper_reset_command_state
    reset_commands = np.repeat(reset_command[None, :], gripper_reset_steps, axis=0).astype(np.float32, copy=False)
    if dry_run:
        logging.info("Dry run: suppressing initial gripper reset with actions=%s", reset_commands.shape)
        return False

    robot_ws.send(
        spiritai_bridge.pack_robot_server_message(
            {
                "type": "send_command",
                "kind": robot_command_kind,
                "actions": reset_commands,
                "source_hz": source_hz,
            }
        )
    )
    ack = _wait_for_robot_command_ack(
        robot_ws,
        timeout_s=command_ack_timeout_s,
        operation="initial gripper reset",
    )
    if not ack.get("accepted", False):
        raise spiritai_bridge.RobotServerProtocolError(f"Initial gripper reset rejected: {ack.get('error')}")
    logging.info(
        "Initial gripper reset accepted: chunk_id=%s actions=%s expected_finish_at=%s",
        ack.get("chunk_id"),
        reset_commands.shape,
        ack.get("expected_finish_at"),
    )
    return True


def _infer_policy_chunk(
    policy: _websocket_client_policy.WebsocketClientPolicy,
    obs: Mapping[str, object],
    images: Mapping[str, object],
    *,
    prompt: str,
    policy_action_layout: Literal["joint", "cartesian"],
    rtc: dict | None = None,
    return_model_actions: bool = False,
) -> PolicyChunk:
    """Run one policy inference and retain model-space and robot-facing actions separately."""
    if policy_action_layout == "cartesian":
        policy_obs = spiritai_bridge.map_robot_server_cartesian_observation(obs, images, prompt=prompt)
    else:
        policy_obs = spiritai_bridge.map_robot_server_observation(obs, images, prompt=prompt)
    result = policy.infer(policy_obs, rtc=rtc, return_model_actions=return_model_actions)
    return PolicyChunk(
        actions=result["actions"],
        model_actions=result.get("model_actions") if return_model_actions else None,
    )


def _action_plan_from_chunk(
    chunk: PolicyChunk,
    *,
    generation_tick: int,
    rtc_metadata: RTCRuntimeMetadata,
) -> ActionPlan:
    if chunk.model_actions is None:
        raise ValueError("Policy did not return model_actions for training-time RTC.")
    model_actions = np.asarray(chunk.model_actions, dtype=np.float32)
    robot_actions = np.asarray(chunk.actions, dtype=np.float32)
    expected_model_shape = (rtc_metadata.action_horizon, rtc_metadata.action_dim)
    if model_actions.shape != expected_model_shape:
        raise ValueError(f"Policy model_actions must have shape {expected_model_shape}, got {model_actions.shape}")
    if robot_actions.ndim != 2 or robot_actions.shape[0] != rtc_metadata.action_horizon:
        raise ValueError(
            "Policy robot-facing actions must be rank 2 with horizon "
            f"{rtc_metadata.action_horizon}, got {robot_actions.shape}"
        )
    if robot_actions.shape[1] <= 0:
        raise ValueError("Policy robot-facing actions must have a positive action dimension.")
    if not np.all(np.isfinite(model_actions)) or not np.all(np.isfinite(robot_actions)):
        raise ValueError("Policy actions must contain only finite values.")
    return ActionPlan(
        generation_tick=generation_tick,
        model_actions=model_actions,
        robot_actions=robot_actions,
    )


def _infer_rtc_action_plan(
    policy: _websocket_client_policy.WebsocketClientPolicy,
    task: RTCInferenceTask,
    *,
    prompt: str,
    policy_action_layout: Literal["joint", "cartesian"],
    rtc_metadata: RTCRuntimeMetadata,
) -> ActionPlan:
    """Worker-only RTC inference that returns a replacement plan at the request start tick."""
    request = task.request
    chunk = _infer_policy_chunk(
        policy,
        task.obs,
        task.images,
        prompt=prompt,
        policy_action_layout=policy_action_layout,
        rtc={
            "algorithm": "training_time_v1",
            "action_prefix": request.action_prefix,
            "delay_steps": request.planned_delay_steps,
        },
        return_model_actions=True,
    )
    return _action_plan_from_chunk(
        chunk,
        generation_tick=request.start_tick,
        rtc_metadata=rtc_metadata,
    )


class PolicyRTCWorker:
    """Own one policy connection exclusively from the RTC inference worker thread."""

    def __init__(
        self,
        *,
        runtime: RuntimeConfig,
        prompt: str,
        policy_action_layout: Literal["joint", "cartesian"],
        policy_factory: Callable[[], _websocket_client_policy.WebsocketClientPolicy] | None = None,
    ) -> None:
        self._runtime = runtime
        self._prompt = prompt
        self._policy_action_layout = policy_action_layout
        self._policy_factory = policy_factory or self._default_policy_factory
        self._policy: _websocket_client_policy.WebsocketClientPolicy | None = None
        self._policy_lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._transport_closed = False
        self._rtc_metadata: RTCRuntimeMetadata | None = None
        self._worker: RTCInferenceWorker[PolicyWorkerTask, PolicyWorkerResult] = RTCInferenceWorker(self._infer)

    def submit(self, task: PolicyWorkerTask) -> Future[RTCInferenceResult[PolicyWorkerResult]]:
        return self._worker.submit(task)

    def close(self, *, wait: bool = True) -> None:
        self._cancel_event.set()
        self._close_policy_transport()
        self._worker.close(wait=wait)

    def _default_policy_factory(self) -> _websocket_client_policy.WebsocketClientPolicy:
        return _websocket_client_policy.WebsocketClientPolicy(
            host=self._runtime.policy.host,
            port=self._runtime.policy.port,
            cancel_event=self._cancel_event,
            connect_timeout_s=self._runtime.policy.connect_timeout_s,
        )

    def _infer(self, task: PolicyWorkerTask) -> PolicyWorkerResult:
        with self._policy_lock:
            policy = self._policy
        if policy is None:
            policy = self._policy_factory()
            with self._policy_lock:
                self._policy = policy
        if self._cancel_event.is_set():
            self._close_policy_transport()
            raise RuntimeError("Policy RTC worker was cancelled before policy inference started.")

        if isinstance(task, PolicyBootstrapTask):
            if self._rtc_metadata is None:
                self._rtc_metadata = validate_rtc_runtime_metadata(self._runtime, policy.get_server_metadata())
            return self._rtc_metadata

        if self._rtc_metadata is None:
            raise RuntimeError("Policy RTC worker must be bootstrapped before inference.")
        if isinstance(task, InitialInferenceTask):
            chunk = _infer_policy_chunk(
                policy,
                task.obs,
                task.images,
                prompt=self._prompt,
                policy_action_layout=self._policy_action_layout,
                rtc=None,
                return_model_actions=True,
            )
            return _action_plan_from_chunk(
                chunk,
                generation_tick=0,
                rtc_metadata=self._rtc_metadata,
            )
        return _infer_rtc_action_plan(
            policy,
            task,
            prompt=self._prompt,
            policy_action_layout=self._policy_action_layout,
            rtc_metadata=self._rtc_metadata,
        )

    def _close_policy_transport(self) -> None:
        with self._policy_lock:
            policy = self._policy
            if policy is None or self._transport_closed:
                return
            self._transport_closed = True
        close = getattr(policy, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logging.exception("Ignoring policy transport close failure during RTC worker shutdown.")


@contextlib.contextmanager
def _robot_session_with_policy_cleanup(
    robot_connection: contextlib.AbstractContextManager[object],
    policy_worker: PolicyRTCWorker,
    *,
    wait_for_policy: Callable[[], bool],
) -> Iterator[object]:
    """Close robot transport before the policy worker can wait on a late RPC."""
    with contextlib.ExitStack() as cleanup:
        cleanup.callback(lambda: policy_worker.close(wait=wait_for_policy()))
        robot_ws = cleanup.enter_context(robot_connection)
        yield robot_ws


def _policy_worker_cleanup_can_wait(
    *,
    bootstrap_future: Future[object] | None,
    initial_inference_future: Future[object] | None,
    flight: RTCInferenceFlight | None,
) -> bool:
    """Return whether every policy operation is complete before shutdown waits."""
    return all(
        future is None or future.done()
        for future in (
            bootstrap_future,
            initial_inference_future,
            None if flight is None else flight.future,
        )
    )


def _wait_for_policy_worker_result(
    future: Future[RTCInferenceResult[PolicyWorkerResult]],
    *,
    timeout_s: float,
    operation: str,
) -> RTCInferenceResult[PolicyWorkerResult]:
    """Bound a synchronous policy-worker operation and report a clear stage name."""
    try:
        return future.result(timeout=timeout_s)
    except FutureTimeoutError as exc:
        raise RuntimeError(f"{operation} timed out after {timeout_s:g}s") from exc


def _current_command_state(
    obs: Mapping[str, object],
    *,
    policy_action_layout: Literal["joint", "cartesian"],
    command_dim: int,
) -> np.ndarray:
    if policy_action_layout == "cartesian":
        return spiritai_bridge.robot_server_obs_to_cartesian_command_layout(obs, command_dim)
    return spiritai_bridge.robot_server_obs_to_joint_command_layout(obs, command_dim)


def _one_row_command_for_dispatch(
    *,
    dispatch_kind: str,
    robot_action: np.ndarray | None,
    current_state: np.ndarray,
    policy_action_layout: Literal["joint", "cartesian"],
    command_dim: int,
) -> np.ndarray:
    if dispatch_kind == "hold":
        return current_state[None, :].astype(np.float32, copy=True)
    if robot_action is None:
        raise RuntimeError("RTC action dispatch is missing its robot-facing action.")

    robot_actions = np.asarray(robot_action, dtype=np.float32).reshape(1, -1)
    if policy_action_layout == "cartesian":
        command = spiritai_bridge.spiritai_cartesian_actions_to_cartesian_commands(robot_actions, command_dim)
    else:
        command = spiritai_bridge.spiritai_actions_to_joint_commands(robot_actions, command_dim)
    if command.shape != (1, command_dim):
        raise RuntimeError(f"RTC command must have shape (1, {command_dim}), got {command.shape}")
    return command


def _apply_one_row_safety_filters(
    command: np.ndarray,
    *,
    dispatch_kind: str,
    current_state: np.ndarray,
    policy_action_layout: Literal["joint", "cartesian"],
    source_hz: float,
    rollback_guard_steps: int,
    rollback_scale: float,
    blend_steps: int,
    motion_limits: spiritai_bridge.JointMotionLimits,
    cartesian_motion_limits: spiritai_bridge.CartesianMotionLimits,
    previous_commands: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Run legacy rollback/blend/limit protections without growing a one-tick command."""
    if command.ndim != 2 or command.shape[0] != 1:
        raise ValueError(f"RTC safety filters require exactly one command, got {command.shape}")
    history = None if dispatch_kind == "hold" else previous_commands
    if policy_action_layout == "cartesian":
        filtered = spiritai_bridge.suppress_cartesian_chunk_start_rollback(
            command,
            history,
            guard_steps=rollback_guard_steps,
            rollback_scale=rollback_scale,
        )
    else:
        filtered = spiritai_bridge.suppress_chunk_start_rollback(
            command,
            history,
            guard_steps=rollback_guard_steps,
            rollback_scale=rollback_scale,
        )

    safe_blend_steps = min(max(blend_steps, 0), max(filtered.shape[0] - 1, 0))
    filtered = spiritai_bridge.blend_joint_command_start(filtered, current_state, safe_blend_steps)
    if policy_action_layout == "cartesian":
        return spiritai_bridge.limit_cartesian_command_motion(
            filtered,
            current_state,
            source_hz=source_hz,
            limits=cartesian_motion_limits,
            previous_commands=history,
        )
    return spiritai_bridge.limit_joint_command_motion(
        filtered,
        current_state,
        source_hz=source_hz,
        limits=motion_limits,
        previous_commands=history,
    )


def _hold_then_stop_reason(
    *,
    worker_misses: int,
    rpc_budget_misses: int,
    max_consecutive: int,
    action: str,
) -> str | None:
    """Return the reason that requires one hold command before stopping."""
    if action != "hold_then_stop":
        return None
    stop_after = max(1, max_consecutive)
    if worker_misses >= stop_after:
        return "worker inference failed or missed its deadline"
    if rpc_budget_misses >= stop_after:
        return "RPC budget exceeded"
    return None


def _dispatch_for_stop_or_plan(
    controller: RTCController,
    *,
    current_tick: int,
    stop_reason: str | None,
) -> tuple[DispatchAction, str | None]:
    """Choose a one-tick hold at the threshold instead of dispatching a stale plan action."""
    if stop_reason is not None:
        return DispatchAction(kind="hold"), stop_reason
    return controller.action_for_tick(current_tick), None


def main(args: BootstrapArgs) -> None:
    config_path = args.config.expanduser().resolve()
    runtime = load_runtime_config(config_path)
    logging.info(
        "RTC runtime config: path=%s robot_url=%s action_layout=%s rtc_mode=%s source_hz=%.3f",
        config_path,
        runtime.robot.url,
        runtime.robot.action_layout,
        runtime.rtc.mode,
        runtime.control.source_hz,
    )

    policy_action_layout: Literal["joint", "cartesian"] = runtime.robot.action_layout  # type: ignore[assignment]
    policy_worker = PolicyRTCWorker(
        runtime=runtime,
        prompt=runtime.policy.prompt,
        policy_action_layout=policy_action_layout,
    )
    bootstrap_future: Future[RTCInferenceResult[PolicyWorkerResult]] | None = None
    initial_inference_future: Future[RTCInferenceResult[PolicyWorkerResult]] | None = None
    flight: RTCInferenceFlight | None = None
    try:
        bootstrap_future = policy_worker.submit(PolicyBootstrapTask())
        bootstrap_result = _wait_for_policy_worker_result(
            bootstrap_future,
            timeout_s=runtime.rtc.initial_inference_timeout_s,
            operation="policy bootstrap metadata",
        ).value
    except Exception:
        policy_worker.close(
            wait=_policy_worker_cleanup_can_wait(
                bootstrap_future=bootstrap_future,
                initial_inference_future=initial_inference_future,
                flight=flight,
            )
        )
        raise
    finally:
        if bootstrap_future is not None and bootstrap_future.done():
            bootstrap_future = None
    if not isinstance(bootstrap_result, RTCRuntimeMetadata):
        policy_worker.close(
            wait=_policy_worker_cleanup_can_wait(
                bootstrap_future=bootstrap_future,
                initial_inference_future=initial_inference_future,
                flight=flight,
            )
        )
        raise RuntimeError("Policy RTC worker returned an invalid bootstrap result.")
    rtc_metadata = bootstrap_result
    logging.info(
        "Policy RTC capability: algorithm=training_time_v1 horizon=%d action_dim=%d training_max_delay_steps=%d",
        rtc_metadata.action_horizon,
        rtc_metadata.action_dim,
        rtc_metadata.training_max_delay_steps,
    )

    with _robot_session_with_policy_cleanup(
        _open_robot_connection(
            runtime.robot.url,
            timeout_s=runtime.control.command_ack_timeout_s,
        ),
        policy_worker,
        wait_for_policy=lambda: _policy_worker_cleanup_can_wait(
            bootstrap_future=bootstrap_future,
            initial_inference_future=initial_inference_future,
            flight=flight,
        ),
    ) as robot_ws:
        hello = _recv_robot_response(
            robot_ws,
            timeout_s=runtime.control.command_ack_timeout_s,
            operation="robot hello",
        )
        if hello.get("type") != "hello":
            raise spiritai_bridge.RobotServerProtocolError(
                f"Expected hello from robot_server, got: {hello.get('type')}"
            )

        robot_metadata = hello["metadata"]
        if policy_action_layout == "cartesian":
            command_dim = spiritai_bridge.choose_cartesian_command_dim(robot_metadata)
            robot_command_kind = "cart"
            summarize_delta = spiritai_bridge.summarize_cartesian_delta_by_group
        else:
            command_dim = spiritai_bridge.choose_joint_command_dim(robot_metadata)
            robot_command_kind = "joint"
            summarize_delta = spiritai_bridge.summarize_joint_delta_by_group
        logging.info(
            "Robot server connected: structure=%s layout=%s command_kind=%s command_dim=%s cart_dim=%s "
            "joint_dim=%s cameras=%s",
            robot_metadata.get("structure"),
            policy_action_layout,
            robot_command_kind,
            command_dim,
            robot_metadata.get("cart_dim"),
            robot_metadata.get("joint_dim"),
            robot_metadata.get("cameras"),
        )
        if runtime.robot.enable_external_following:
            _set_robot_external_following(
                robot_ws,
                enabled=True,
                timeout_s=runtime.control.command_ack_timeout_s,
            )
        if runtime.control.startup_delay_s > 0:
            logging.info("Startup delay %.1fs before first inference.", runtime.control.startup_delay_s)
            time.sleep(runtime.control.startup_delay_s)

        max_chunk = int(robot_metadata.get("max_chunk", 60))
        if runtime.robot.gripper_reset_steps > max_chunk:
            raise ValueError(
                "gripper_reset_steps must be <= robot_server max_chunk "
                f"({max_chunk}), got {runtime.robot.gripper_reset_steps}"
            )
        motion_limits = spiritai_bridge.JointMotionLimits(
            max_arm_velocity_rad_s=runtime.control.motion_limits.max_arm_velocity_rad_s,
            max_torso_velocity_rad_s=runtime.control.motion_limits.max_torso_velocity_rad_s,
            max_gripper_velocity_s=runtime.control.motion_limits.max_gripper_velocity_s,
            max_base_speed=runtime.control.motion_limits.max_base_speed,
            max_joint_accel_rad_s2=runtime.control.motion_limits.max_joint_accel_rad_s2,
        )
        cartesian_motion_limits = spiritai_bridge.CartesianMotionLimits(
            max_arm_translation_m_s=runtime.control.motion_limits.max_cart_translation_m_s,
            max_arm_rotation_rad_s=runtime.control.motion_limits.max_cart_rotation_rad_s,
            max_torso_translation_m_s=runtime.control.motion_limits.max_torso_cart_translation_m_s,
            max_torso_rotation_rad_s=runtime.control.motion_limits.max_torso_cart_rotation_rad_s,
            max_gripper_velocity_s=runtime.control.motion_limits.max_gripper_velocity_s,
            max_base_speed=runtime.control.motion_limits.max_base_speed,
            max_cart_accel=runtime.control.motion_limits.max_cart_accel,
        )

        initial_preflight_started_at = time.perf_counter()
        _wait_until_robot_idle(
            robot_ws,
            runtime.control.busy_sleep_s,
            timeout_s=runtime.control.command_ack_timeout_s,
            idle_timeout_s=runtime.control.robot_idle_timeout_s,
        )
        initial_obs, initial_images = _get_robot_obs(
            robot_ws,
            timeout_s=runtime.control.command_ack_timeout_s,
        )
        logging.info("Initial robot read-only preflight latency: %.3fs", time.perf_counter() - initial_preflight_started_at)
        left_gripper, right_gripper = _get_gripper_state(initial_obs)
        if _grippers_at_initial_state(
            initial_obs,
            initial_gripper_obs_state=runtime.robot.initial_gripper_obs_state,
            tolerance=runtime.robot.gripper_initial_tolerance,
        ):
            logging.info(
                "Initial gripper state OK: left=%.4f right=%.4f target=%.4f",
                left_gripper,
                right_gripper,
                runtime.robot.initial_gripper_obs_state,
            )
        else:
            logging.warning(
                "Initial gripper state is not reset: left=%.4f right=%.4f target=%.4f tolerance=%.4f; "
                "resetting before inference with command=%.4f.",
                left_gripper,
                right_gripper,
                runtime.robot.initial_gripper_obs_state,
                runtime.robot.gripper_initial_tolerance,
                runtime.robot.gripper_reset_command_state,
            )
            reset_sent = _send_initial_gripper_reset(
                robot_ws,
                initial_obs,
                policy_action_layout=policy_action_layout,
                robot_command_kind=robot_command_kind,
                command_dim=command_dim,
                gripper_reset_command_state=runtime.robot.gripper_reset_command_state,
                gripper_reset_steps=runtime.robot.gripper_reset_steps,
                source_hz=runtime.control.source_hz,
                command_ack_timeout_s=runtime.control.command_ack_timeout_s,
                dry_run=args.dry_run,
            )
            if reset_sent:
                verify_preflight_started_at = time.perf_counter()
                _wait_until_robot_idle(
                    robot_ws,
                    runtime.control.busy_sleep_s,
                    timeout_s=runtime.control.command_ack_timeout_s,
                    idle_timeout_s=runtime.control.robot_idle_timeout_s,
                )
                initial_obs, initial_images = _get_robot_obs(
                    robot_ws,
                    timeout_s=runtime.control.command_ack_timeout_s,
                )
                logging.info(
                    "Initial reset verification read-only latency: %.3fs",
                    time.perf_counter() - verify_preflight_started_at,
                )
                verify_left_gripper, verify_right_gripper = _get_gripper_state(initial_obs)
                if not _grippers_at_initial_state(
                    initial_obs,
                    initial_gripper_obs_state=runtime.robot.initial_gripper_obs_state,
                    tolerance=runtime.robot.gripper_initial_tolerance,
                ):
                    raise RuntimeError(
                        "Initial gripper reset did not reach target: "
                        f"left={verify_left_gripper:.4f} right={verify_right_gripper:.4f} "
                        f"target={runtime.robot.initial_gripper_obs_state:.4f} "
                        f"tolerance={runtime.robot.gripper_initial_tolerance:.4f}"
                    )

        initial_inference_started_at = time.perf_counter()
        initial_inference_future = policy_worker.submit(InitialInferenceTask(obs=initial_obs, images=initial_images))
        try:
            initial_result = _wait_for_policy_worker_result(
                initial_inference_future,
                timeout_s=runtime.rtc.initial_inference_timeout_s,
                operation="initial policy inference",
            )
        finally:
            if initial_inference_future.done():
                initial_inference_future = None
        logging.info(
            "Initial worker policy inference latency: %.3fs", time.perf_counter() - initial_inference_started_at
        )
        if not isinstance(initial_result.value, ActionPlan):
            raise RuntimeError("Policy RTC worker returned an invalid initial action plan.")
        initial_plan = initial_result.value
        controller = RTCController(
            action_horizon=rtc_metadata.action_horizon,
            action_dim=rtc_metadata.action_dim,
            s_min=runtime.rtc.s_min,
            training_max_delay_steps=rtc_metadata.training_max_delay_steps,
        )
        controller.install_initial_plan(initial_plan)

        request_budget_blocked = False
        previous_state: np.ndarray | None = None
        accepted_command_history: deque[np.ndarray] = deque(maxlen=2)
        accepted_ack_durations: deque[float] = deque(maxlen=runtime.rtc.delay.history_window)
        consecutive_rpc_budget_misses = 0
        rpc_budget_s = (1.0 / runtime.control.source_hz) * runtime.control.rpc_budget_fraction

        try:
            for _loop_step in range(runtime.control.max_steps):
                read_only_started_at = time.perf_counter()
                _wait_until_robot_idle(
                    robot_ws,
                    runtime.control.busy_sleep_s,
                    timeout_s=runtime.control.command_ack_timeout_s,
                    idle_timeout_s=runtime.control.robot_idle_timeout_s,
                )
                obs, images = _get_robot_obs(
                    robot_ws,
                    timeout_s=runtime.control.command_ack_timeout_s,
                )
                read_only_duration_s = time.perf_counter() - read_only_started_at
                current_tick = controller.accepted_tick
                current_state = _current_command_state(
                    obs,
                    policy_action_layout=policy_action_layout,
                    command_dim=command_dim,
                )
                if previous_state is not None:
                    actual_delta_summary = summarize_delta(current_state - previous_state, command_dim)
                    logging.info(
                        "Tick %d actual state delta: %s",
                        current_tick,
                        " ".join(
                            f"{name}=mean:{mean_abs:.4f}/max:{max_abs:.4f}"
                            for name, (mean_abs, max_abs) in actual_delta_summary.items()
                        ),
                    )

                rolling_ack_duration_s = max(accepted_ack_durations, default=0.0)
                rpc_budget_exceeded = read_only_duration_s + rolling_ack_duration_s > rpc_budget_s
                if rpc_budget_exceeded:
                    consecutive_rpc_budget_misses += 1
                    logging.warning(
                        "Tick %d RPC budget exceeded: read_only=%.3fs rolling_accepted_ack=%.3fs budget=%.3fs",
                        current_tick,
                        read_only_duration_s,
                        rolling_ack_duration_s,
                        rpc_budget_s,
                    )
                    if flight is not None and not flight.expired:
                        request_budget_blocked = True
                else:
                    consecutive_rpc_budget_misses = 0
                logging.info("Tick %d read-only preflight latency: %.3fs", current_tick, read_only_duration_s)

                if flight is not None:
                    if flight.future.done():
                        completed_flight = flight
                        flight = None
                        try:
                            result = completed_flight.completed_result()
                        except Exception:
                            if completed_flight.expired:
                                logging.exception("Tick %d discarding failed expired RTC worker result", current_tick)
                            else:
                                logging.exception("Tick %d RTC worker inference failed", current_tick)
                                controller.record_failed_request(completed_flight.request)
                        else:
                            if result is None:
                                logging.warning("Tick %d discarding expired RTC worker result", current_tick)
                            else:
                                logging.info(
                                    "Tick %d RTC worker inference latency: %.3fs",
                                    current_tick,
                                    result.finished_at - result.started_at,
                                )
                                if request_budget_blocked:
                                    logging.error(
                                        "Tick %d discarding RTC plan because its request crossed the RPC budget",
                                        current_tick,
                                    )
                                    controller.record_failed_request(completed_flight.request)
                                else:
                                    accepted = controller.accept_result(
                                        completed_flight.request,
                                        result.value,
                                        completion_tick=current_tick,
                                    )
                                    if not accepted:
                                        logging.warning(
                                            "Tick %d RTC plan missed its deadline and was not installed", current_tick
                                        )
                        request_budget_blocked = False
                    elif flight.expire_if_due(controller, current_tick=current_tick):
                        request_budget_blocked = False
                        logging.warning(
                            "Tick %d RTC request=%d expired before worker completion",
                            current_tick,
                            flight.request.request_id,
                        )
                    elif flight.expired:
                        # The sole executor still owns the policy WebSocket. A retry cannot begin
                        # until this stale call returns; issuing one now would create a concurrent
                        # policy query. Count the blocked tick so a hung worker reaches hold/stop.
                        controller.record_worker_unavailable()
                        logging.error(
                            "Tick %d expired RTC request=%d still blocks the single policy worker",
                            current_tick,
                            flight.request.request_id,
                        )

                stop_reason = _hold_then_stop_reason(
                    worker_misses=controller.consecutive_deadline_misses,
                    rpc_budget_misses=consecutive_rpc_budget_misses,
                    max_consecutive=runtime.rtc.deadline_miss.max_consecutive,
                    action=runtime.rtc.deadline_miss.action,
                )
                if stop_reason is not None:
                    logging.error("Tick %d threshold reached (%s); sending one hold before stopping", current_tick, stop_reason)

                active_plan = controller.current_plan
                request_due = (
                    active_plan is not None
                    and current_tick
                    >= active_plan.generation_tick + max(runtime.rtc.delay.planned_max_steps, runtime.rtc.s_min)
                    and current_tick + runtime.rtc.delay.planned_max_steps
                    <= active_plan.generation_tick + rtc_metadata.action_horizon
                )
                if (
                    stop_reason is None
                    and not rpc_budget_exceeded
                    and flight is None
                    and controller.inflight_request is None
                    and request_due
                ):
                    request = controller.start_request(
                        current_tick=current_tick,
                        planned_delay_steps=runtime.rtc.delay.planned_max_steps,
                    )
                    flight = RTCInferenceFlight(
                        request=request,
                        future=policy_worker.submit(RTCInferenceTask(request=request, obs=obs, images=images)),
                    )
                    logging.info(
                        "Tick %d submitted RTC request=%d delay_steps=%d",
                        current_tick,
                        request.request_id,
                        request.planned_delay_steps,
                    )

                dispatch, stop_after_dispatch = _dispatch_for_stop_or_plan(
                    controller,
                    current_tick=current_tick,
                    stop_reason=stop_reason,
                )
                command = _one_row_command_for_dispatch(
                    dispatch_kind=dispatch.kind,
                    robot_action=dispatch.robot_action,
                    current_state=current_state,
                    policy_action_layout=policy_action_layout,
                    command_dim=command_dim,
                )
                previous_commands = (
                    np.stack(tuple(accepted_command_history), axis=0) if accepted_command_history else None
                )
                command, motion_stats = _apply_one_row_safety_filters(
                    command,
                    dispatch_kind=dispatch.kind,
                    current_state=current_state,
                    policy_action_layout=policy_action_layout,
                    source_hz=runtime.control.source_hz,
                    rollback_guard_steps=runtime.control.rollback_guard_steps,
                    rollback_scale=runtime.control.rollback_scale,
                    blend_steps=runtime.control.blend_steps,
                    motion_limits=motion_limits,
                    cartesian_motion_limits=cartesian_motion_limits,
                    previous_commands=previous_commands,
                )
                logging.info(
                    "Tick %d dispatch=%s command=%s motion_limit=raw_max_vel:%.4f limited_max_vel:%.4f "
                    "limited_fraction:%.2f",
                    current_tick,
                    dispatch.kind,
                    command.shape,
                    motion_stats["max_raw_velocity"],
                    motion_stats["max_limited_velocity"],
                    motion_stats["limited_fraction"],
                )

                if args.dry_run:
                    logging.info("Tick %d dry run: suppressing one %s command", current_tick, dispatch.kind)
                    if stop_after_dispatch is not None:
                        raise RuntimeError(
                            f"Stopping after {stop_after_dispatch}; dry run suppressed the one-row hold command."
                        )
                    continue

                ack_started_at = time.perf_counter()
                robot_ws.send(
                    spiritai_bridge.pack_robot_server_message(
                        {
                            "type": "send_command",
                            "kind": robot_command_kind,
                            "actions": command,
                            "source_hz": runtime.control.source_hz,
                        }
                    )
                )
                ack = _wait_for_robot_command_ack(
                    robot_ws,
                    timeout_s=runtime.control.command_ack_timeout_s,
                    operation="terminal hold" if stop_after_dispatch is not None else "robot command",
                )
                ack_duration_s = time.perf_counter() - ack_started_at
                if not ack.get("accepted", False):
                    logging.warning(
                        "Tick %d rejected by robot_server after %.3fs: %s",
                        current_tick,
                        ack_duration_s,
                        ack.get("error"),
                    )
                    if stop_after_dispatch is not None:
                        raise RuntimeError(
                            f"Stopping after {stop_after_dispatch}; robot rejected the one-row hold command."
                        )
                    continue

                controller.record_accepted_tick(acknowledged=True)
                accepted_ack_durations.append(ack_duration_s)
                previous_state = current_state.copy()
                accepted_command_history.append(command[0].copy())
                logging.info(
                    "Tick %d accepted: chunk_id=%s actions=%s expected_finish_at=%s ack_latency=%.3fs",
                    current_tick,
                    ack.get("chunk_id"),
                    command.shape,
                    ack.get("expected_finish_at"),
                    ack_duration_s,
                )
                if stop_after_dispatch is not None:
                    raise RuntimeError(
                        f"Stopping after {stop_after_dispatch}; acknowledged one-row hold command was sent."
                    )
        finally:
            logging.info("RTC control loop exiting; robot transport closes before policy worker cleanup.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(BootstrapArgs))
