"""Tests for the SpiritAI training-time RTC entrypoint."""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from openpi.rtc import RTCRequest

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


def test_worker_inference_sends_the_training_time_envelope_and_builds_start_tick_plan():
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

    class FakePolicy:
        def __init__(self):
            self.calls = []

        def infer(self, policy_obs, *, rtc, return_model_actions):
            self.calls.append((policy_obs, rtc, return_model_actions))
            return {
                "actions": np.full((4, 27), 5.0, dtype=np.float32),
                "model_actions": np.arange(8, dtype=np.float32).reshape(4, 2),
            }

    policy = FakePolicy()
    task = main.RTCInferenceTask(request=request, obs=obs, images=images)
    metadata = main.RTCRuntimeMetadata(action_horizon=4, action_dim=2, training_max_delay_steps=2)

    plan = main.infer_rtc_action_plan(
        policy,
        task,
        prompt="fold the paper box",
        policy_action_layout="joint",
        rtc_metadata=metadata,
    )

    assert len(policy.calls) == 1
    policy_obs, rtc, return_model_actions = policy.calls[0]
    assert policy_obs["prompt"] == "fold the paper box"
    assert rtc["algorithm"] == "training_time_v1"
    np.testing.assert_array_equal(rtc["action_prefix"], action_prefix)
    assert rtc["delay_steps"] == 2
    assert return_model_actions is True
    assert plan.generation_tick == request.start_tick
    assert plan.model_actions.shape == (4, 2)
    assert plan.robot_actions.shape == (4, 27)


def test_bootstrap_args_is_a_frozen_dataclass():
    main = load_main_module()

    assert dataclasses.is_dataclass(main.BootstrapArgs)
    with pytest.raises(dataclasses.FrozenInstanceError):
        main.BootstrapArgs().dry_run = True
