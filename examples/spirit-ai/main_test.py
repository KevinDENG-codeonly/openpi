"""Tests for the SpiritAI training-time RTC entrypoint."""

from __future__ import annotations

from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
import dataclasses
import importlib.util
import inspect
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from openpi.rtc import ActionPlan
from openpi.rtc import RTCController
from openpi.rtc import RTCRequest
from openpi.rtc.worker import RTCInferenceResult

MAIN_PATH = Path(__file__).with_name("main.py")


def load_main_module():
    """Load the example entrypoint without making its directory a package."""
    module_name = f"spirit_ai_main_test_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, MAIN_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


def make_runtime(*, planned_max_steps: int = 2):
    return SimpleNamespace(
        rtc=SimpleNamespace(delay=SimpleNamespace(planned_max_steps=planned_max_steps)),
    )


def make_policy_metadata(*, horizon: int = 4, action_dim: int = 2, max_delay_steps: int = 2):
    return {
        "rtc_capabilities": {
            "algorithm": "training_time_v1",
            "action_horizon": horizon,
            "action_dim": action_dim,
            "training_max_delay_steps": max_delay_steps,
        }
    }


def make_joint_observation() -> tuple[dict, dict]:
    obs = {
        "leftarm_state_joint_pos": np.zeros(7, dtype=np.float32),
        "leftarm_state_psi": np.zeros(1, dtype=np.float32),
        "leftarm_gripper_state_pos": np.zeros(1, dtype=np.float32),
        "rightarm_state_joint_pos": np.zeros(7, dtype=np.float32),
        "rightarm_state_psi": np.zeros(1, dtype=np.float32),
        "rightarm_gripper_state_pos": np.zeros(1, dtype=np.float32),
        "torso_state_joint_pos": np.zeros(6, dtype=np.float32),
        "base_state_speed": np.zeros(3, dtype=np.float32),
    }
    images = {
        "cam_high": np.zeros((2, 2, 3), dtype=np.uint8),
        "cam_left_wrist": np.zeros((2, 2, 3), dtype=np.uint8),
        "cam_right_wrist": np.zeros((2, 2, 3), dtype=np.uint8),
    }
    return obs, images


def test_bootstrap_defaults_use_the_source_relative_runtime_config():
    main = load_main_module()

    expected_config = MAIN_PATH.parent / "configs" / "rtc" / "training_time.yaml"
    assert expected_config == main.DEFAULT_RUNTIME_CONFIG
    assert main.BootstrapArgs() == main.BootstrapArgs(config=expected_config, dry_run=False)
    assert main.BootstrapArgs.__dataclass_params__.frozen is True
    assert not hasattr(main, "Args")


def test_importing_main_does_not_read_runtime_config(monkeypatch):
    original_read_text = Path.read_text

    def fail_if_yaml_is_read(path: Path, *args, **kwargs):
        if path.suffix == ".yaml":
            raise AssertionError(f"main import unexpectedly read {path}")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_if_yaml_is_read)

    load_main_module()


def test_validate_rtc_runtime_metadata_returns_model_dimensions():
    main = load_main_module()

    capability = main.validate_rtc_runtime_metadata(make_runtime(), make_policy_metadata())

    assert capability.action_horizon == 4
    assert capability.action_dim == 2
    assert capability.training_max_delay_steps == 2


@pytest.mark.parametrize(
    ("runtime", "metadata", "message"),
    [
        (make_runtime(planned_max_steps=3), make_policy_metadata(max_delay_steps=2), "planned_max_steps"),
        (make_runtime(), {}, "rtc_capabilities"),
        (make_runtime(), {"rtc_capabilities": {"algorithm": "disabled"}}, "algorithm"),
        (make_runtime(), make_policy_metadata(horizon=0), "action_horizon"),
        (make_runtime(), make_policy_metadata(action_dim=True), "action_dim"),
        (make_runtime(), make_policy_metadata(max_delay_steps=-1), "training_max_delay_steps"),
    ],
)
def test_validate_rtc_runtime_metadata_rejects_incompatible_capabilities(runtime, metadata, message):
    main = load_main_module()

    with pytest.raises(ValueError, match=message):
        main.validate_rtc_runtime_metadata(runtime, metadata)


def test_policy_worker_owns_connection_metadata_and_all_inference_calls():
    main = load_main_module()
    obs, images = make_joint_observation()
    action_prefix = np.zeros((4, 2), dtype=np.float32)
    action_prefix[:2] = [[10.0, 11.0], [12.0, 13.0]]
    request = RTCRequest(
        request_id=7,
        source_generation_tick=0,
        start_tick=3,
        planned_delay_steps=2,
        execution_horizon=3,
        action_prefix=action_prefix,
        frozen_prefix=action_prefix[:2],
    )

    main_thread_id = threading.get_ident()
    factory_thread_ids = []
    metadata_thread_ids = []
    inference_thread_ids = []

    class FakePolicy:
        def __init__(self):
            self.calls = []

        def get_server_metadata(self):
            metadata_thread_ids.append(threading.get_ident())
            return make_policy_metadata()

        def infer(self, policy_obs, *, rtc, return_model_actions):
            inference_thread_ids.append(threading.get_ident())
            self.calls.append((policy_obs, rtc, return_model_actions))
            return {
                "actions": np.full((4, 27), 5.0, dtype=np.float32),
                "model_actions": np.arange(8, dtype=np.float32).reshape(4, 2),
            }

    policy = FakePolicy()

    def policy_factory():
        factory_thread_ids.append(threading.get_ident())
        return policy

    policy_worker = main.PolicyRTCWorker(
        runtime=make_runtime(),
        prompt="fold the paper box",
        policy_action_layout="joint",
        policy_factory=policy_factory,
    )
    try:
        rtc_metadata = policy_worker.submit(main.PolicyBootstrapTask()).result().value
        initial_plan = policy_worker.submit(main.InitialInferenceTask(obs=obs, images=images)).result().value
        plan = policy_worker.submit(main.RTCInferenceTask(request=request, obs=obs, images=images)).result().value
    finally:
        policy_worker.close()

    assert isinstance(rtc_metadata, main.RTCRuntimeMetadata)
    assert isinstance(initial_plan, ActionPlan)
    assert isinstance(plan, ActionPlan)
    assert len(policy.calls) == 2
    policy_obs, rtc, return_model_actions = policy.calls[1]
    assert policy_obs["prompt"] == "fold the paper box"
    assert rtc["algorithm"] == "training_time_v1"
    np.testing.assert_array_equal(rtc["action_prefix"], action_prefix)
    assert rtc["delay_steps"] == 2
    assert return_model_actions is True
    assert plan.generation_tick == request.start_tick
    assert plan.model_actions.shape == (4, 2)
    assert plan.robot_actions.shape == (4, 27)
    assert len(metadata_thread_ids) == 1
    assert len(factory_thread_ids) == 1
    assert set(factory_thread_ids + metadata_thread_ids + inference_thread_ids).isdisjoint({main_thread_id})
    assert len(set(factory_thread_ids + metadata_thread_ids + inference_thread_ids)) == 1


def test_main_delegates_all_policy_ownership_to_policy_rtc_worker():
    main = load_main_module()
    main_source = inspect.getsource(main.main)

    assert "PolicyRTCWorker(" in main_source
    assert "WebsocketClientPolicy(" not in main_source
    assert ".get_server_metadata(" not in main_source
    assert "_infer_policy_chunk(" not in main_source


def test_expired_slow_worker_result_is_discarded_and_the_active_plan_can_retry():
    main = load_main_module()
    controller = RTCController(action_horizon=8, action_dim=2, s_min=1, training_max_delay_steps=2)
    original_plan = ActionPlan(
        generation_tick=0,
        model_actions=np.arange(16, dtype=np.float32).reshape(8, 2),
        robot_actions=np.zeros((8, 3), dtype=np.float32),
    )
    controller.install_initial_plan(original_plan)
    request = controller.start_request(current_tick=1, planned_delay_steps=1)
    future = Future()
    flight = main.RTCInferenceFlight(request=request, future=future)

    assert flight.expire_if_due(controller, current_tick=1) is False
    assert flight.expire_if_due(controller, current_tick=2) is True
    assert flight.expired is True
    assert controller.current_plan is original_plan
    assert controller.inflight_request is None
    assert controller.deadline_miss_count == 1

    future.set_result(
        RTCInferenceResult(
            value=ActionPlan(
                generation_tick=1,
                model_actions=np.full((8, 2), 99.0, dtype=np.float32),
                robot_actions=np.full((8, 3), 99.0, dtype=np.float32),
            ),
            started_at=1.0,
            finished_at=2.0,
        )
    )
    assert flight.completed_result() is None
    assert controller.current_plan is original_plan

    retry = controller.start_request(current_tick=2, planned_delay_steps=1)
    assert retry.request_id == 1
    assert retry.start_tick == 2
    np.testing.assert_array_equal(retry.frozen_prefix, original_plan.model_actions[2:3])


def test_never_completing_worker_is_expired_and_counts_toward_stop_budget():
    main = load_main_module()
    controller = RTCController(action_horizon=8, action_dim=2, s_min=1, training_max_delay_steps=2)
    controller.install_initial_plan(
        ActionPlan(
            generation_tick=0,
            model_actions=np.zeros((8, 2), dtype=np.float32),
            robot_actions=np.zeros((8, 3), dtype=np.float32),
        )
    )
    request = controller.start_request(current_tick=1, planned_delay_steps=1)
    future = Future()
    flight = main.RTCInferenceFlight(request=request, future=future)

    assert flight.expire_if_due(controller, current_tick=2) is True
    assert future.done() is False
    controller.record_worker_unavailable()

    assert controller.inflight_request is None
    assert controller.deadline_miss_count == 2
    assert controller.consecutive_deadline_misses == 2


def test_robot_session_closes_robot_before_abandoning_a_hung_policy_worker():
    main = load_main_module()
    events = []

    class FakePolicyWorker:
        def close(self, *, wait=True):
            events.append(("policy_close", wait))

    class FakeRobotConnection:
        def __enter__(self):
            events.append(("robot_enter", None))
            return object()

        def __exit__(self, exc_type, exc_value, traceback):
            events.append(("robot_exit", None))

    with main._robot_session_with_policy_cleanup(  # noqa: SLF001
        FakeRobotConnection(),
        FakePolicyWorker(),
        wait_for_policy=lambda: False,
    ):
        events.append(("loop_stop", None))

    assert events == [
        ("robot_enter", None),
        ("loop_stop", None),
        ("robot_exit", None),
        ("policy_close", False),
    ]


def test_synchronous_policy_wait_uses_the_configured_timeout():
    main = load_main_module()
    timeouts = []

    class TimedOutFuture:
        def result(self, *, timeout):
            timeouts.append(timeout)
            raise FutureTimeoutError()

    with pytest.raises(RuntimeError, match="initial policy inference timed out after 0.25s"):
        main._wait_for_policy_worker_result(  # noqa: SLF001
            TimedOutFuture(),
            timeout_s=0.25,
            operation="initial policy inference",
        )

    assert timeouts == [0.25]


def test_robot_session_closes_robot_before_abandoning_timed_out_initial_inference():
    main = load_main_module()
    events = []
    initial_future = Future()

    class FakePolicyWorker:
        def close(self, *, wait=True):
            events.append(("policy_close", wait))

    class FakeRobotConnection:
        def __enter__(self):
            events.append(("robot_enter", None))
            return object()

        def __exit__(self, exc_type, exc_value, traceback):
            events.append(("robot_exit", None))

    with main._robot_session_with_policy_cleanup(  # noqa: SLF001
        FakeRobotConnection(),
        FakePolicyWorker(),
        wait_for_policy=lambda: main._policy_worker_cleanup_can_wait(  # noqa: SLF001
            bootstrap_future=None,
            initial_inference_future=initial_future,
            flight=None,
        ),
    ):
        events.append(("initial_timeout", None))

    assert events == [
        ("robot_enter", None),
        ("initial_timeout", None),
        ("robot_exit", None),
        ("policy_close", False),
    ]


@pytest.mark.parametrize(
    ("worker_misses", "rpc_budget_misses", "expected_reason"),
    [
        (2, 0, "worker inference failed or missed its deadline"),
        (0, 2, "RPC budget exceeded"),
    ],
)
def test_stop_threshold_replaces_a_stale_plan_action_with_a_one_row_hold(
    worker_misses: int,
    rpc_budget_misses: int,
    expected_reason: str,
):
    main = load_main_module()
    controller = RTCController(action_horizon=4, action_dim=2, s_min=1, training_max_delay_steps=2)
    controller.install_initial_plan(
        ActionPlan(
            generation_tick=0,
            model_actions=np.ones((4, 2), dtype=np.float32),
            robot_actions=np.full((4, 3), 9.0, dtype=np.float32),
        )
    )
    reason = main._hold_then_stop_reason(  # noqa: SLF001
        worker_misses=worker_misses,
        rpc_budget_misses=rpc_budget_misses,
        max_consecutive=2,
        action="hold_then_stop",
    )

    dispatch, stop_after_dispatch = main._dispatch_for_stop_or_plan(  # noqa: SLF001
        controller,
        current_tick=0,
        stop_reason=reason,
    )
    command = main._one_row_command_for_dispatch(  # noqa: SLF001
        dispatch_kind=dispatch.kind,
        robot_action=dispatch.robot_action,
        current_state=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        policy_action_layout="joint",
        command_dim=3,
    )

    assert reason == expected_reason
    assert stop_after_dispatch == expected_reason
    assert dispatch.kind == "hold"
    np.testing.assert_array_equal(command, [[1.0, 2.0, 3.0]])


def test_bootstrap_args_is_a_frozen_dataclass():
    main = load_main_module()

    assert dataclasses.is_dataclass(main.BootstrapArgs)
    with pytest.raises(dataclasses.FrozenInstanceError):
        main.BootstrapArgs().dry_run = True
