"""Tests for the strict training-time RTC runtime configuration."""

import copy
import dataclasses
from pathlib import Path

import pytest
import yaml

from openpi.rtc.runtime_config import RuntimeConfigError
from openpi.rtc.runtime_config import default_config_path
from openpi.rtc.runtime_config import load_runtime_config

VALID_RUNTIME = {
    "schema_version": 1,
    "policy": {
        "host": "localhost",
        "port": 8000,
        "prompt": "fold",
        "connect_timeout_s": 1.0,
    },
    "robot": {
        "url": "ws://robot",
        "action_layout": "cartesian",
        "enable_external_following": False,
        "initial_gripper_obs_state": 0.0965,
        "gripper_initial_tolerance": 0.00965,
        "gripper_reset_command_state": 1.0,
        "gripper_reset_steps": 10,
    },
    "control": {
        "source_hz": 15.0,
        "max_steps": 20,
        "busy_sleep_s": 0.01,
        "startup_delay_s": 0.0,
        "blend_steps": 4,
        "rollback_guard_steps": 4,
        "rollback_scale": 0.2,
        "rpc_budget_fraction": 0.7,
        "command_ack_timeout_s": 1.0,
        "motion_limits": {
            "max_arm_velocity_rad_s": 0.35,
            "max_torso_velocity_rad_s": 0.2,
            "max_gripper_velocity_s": 0.8,
            "max_base_speed": 0.05,
            "max_joint_accel_rad_s2": 0.0,
            "max_cart_translation_m_s": 0.08,
            "max_cart_rotation_rad_s": 0.35,
            "max_torso_cart_translation_m_s": 0.04,
            "max_torso_cart_rotation_rad_s": 0.2,
            "max_cart_accel": 0.0,
        },
    },
    "rtc": {
        "mode": "training_time",
        "s_min": 5,
        "initial_inference_timeout_s": 10.0,
        "delay": {"planned_max_steps": 12, "history_window": 16, "safety_margin_steps": 1},
        "deadline_miss": {"max_consecutive": 2, "action": "hold_then_stop"},
    },
}


def write_runtime(path: Path, runtime: dict) -> None:
    path.write_text(yaml.safe_dump(runtime), encoding="utf-8")


def valid_runtime_yaml() -> str:
    return yaml.safe_dump(VALID_RUNTIME)


def test_default_config_path_is_entrypoint_relative_without_reading(tmp_path: Path) -> None:
    entrypoint = tmp_path / "does-not-exist" / "main.py"

    assert default_config_path(entrypoint) == entrypoint.parent / "configs" / "rtc" / "training_time.yaml"


def test_parser_rejects_unknown_top_level_key(tmp_path: Path) -> None:
    runtime = copy.deepcopy(VALID_RUNTIME)
    runtime["unknown"] = True
    path = tmp_path / "bad.yaml"
    write_runtime(path, runtime)

    with pytest.raises(RuntimeConfigError, match="unknown keys"):
        load_runtime_config(path)


def test_parser_rejects_duplicate_root_mapping_key(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text(f"schema_version: 1\n{valid_runtime_yaml()}", encoding="utf-8")

    with pytest.raises(RuntimeConfigError, match="duplicate mapping key 'schema_version'"):
        load_runtime_config(path)


def test_parser_rejects_duplicate_control_mapping_key(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    document = valid_runtime_yaml().replace(
        "  source_hz: 15.0\n",
        "  source_hz: 15.0\n  source_hz: 15.0\n",
    )
    path.write_text(document, encoding="utf-8")

    with pytest.raises(RuntimeConfigError, match="duplicate mapping key 'source_hz'"):
        load_runtime_config(path)


def test_parser_rejects_duplicate_motion_limit_mapping_key(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    document = valid_runtime_yaml().replace(
        "    max_base_speed: 0.05\n",
        "    max_base_speed: 0.05\n    max_base_speed: 0.05\n",
    )
    path.write_text(document, encoding="utf-8")

    with pytest.raises(RuntimeConfigError, match="duplicate mapping key 'max_base_speed'"):
        load_runtime_config(path)


def test_parser_normalizes_unhashable_mapping_key(tmp_path: Path) -> None:
    path = tmp_path / "invalid-key.yaml"
    path.write_text("? [not, hashable]\n: value\n", encoding="utf-8")

    with pytest.raises(RuntimeConfigError, match="unhashable mapping key"):
        load_runtime_config(path)


def test_parser_loads_training_time_profile(tmp_path: Path) -> None:
    path = tmp_path / "runtime.yaml"
    write_runtime(path, VALID_RUNTIME)

    cfg = load_runtime_config(path)

    assert cfg.rtc.mode == "training_time"
    assert cfg.control.source_hz == 15.0
    assert cfg.policy.connect_timeout_s == 1.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.policy.host = "other"


def test_parser_rejects_unknown_nested_key(tmp_path: Path) -> None:
    runtime = copy.deepcopy(VALID_RUNTIME)
    runtime["control"]["motion_limits"]["extra_limit"] = 1.0
    path = tmp_path / "bad.yaml"
    write_runtime(path, runtime)

    with pytest.raises(RuntimeConfigError, match="control.motion_limits: unknown keys"):
        load_runtime_config(path)


def test_parser_rejects_missing_nested_key(tmp_path: Path) -> None:
    runtime = copy.deepcopy(VALID_RUNTIME)
    del runtime["robot"]["url"]
    path = tmp_path / "bad.yaml"
    write_runtime(path, runtime)

    with pytest.raises(RuntimeConfigError, match="robot: missing keys"):
        load_runtime_config(path)


def test_parser_rejects_empty_nested_mapping_as_missing_keys(tmp_path: Path) -> None:
    runtime = copy.deepcopy(VALID_RUNTIME)
    runtime["policy"] = {}
    path = tmp_path / "bad.yaml"
    write_runtime(path, runtime)

    with pytest.raises(RuntimeConfigError, match="policy: missing keys"):
        load_runtime_config(path)


def test_parser_rejects_boolean_for_integer_field(tmp_path: Path) -> None:
    runtime = copy.deepcopy(VALID_RUNTIME)
    runtime["policy"]["port"] = True
    path = tmp_path / "bad.yaml"
    write_runtime(path, runtime)

    with pytest.raises(RuntimeConfigError, match="policy.port must be an integer"):
        load_runtime_config(path)


def test_parser_normalizes_invalid_utf8_document(tmp_path: Path) -> None:
    path = tmp_path / "invalid-utf8.yaml"
    path.write_bytes(b"\x80")

    with pytest.raises(RuntimeConfigError, match="valid UTF-8"):
        load_runtime_config(path)


def test_parser_normalizes_overflowing_yaml_integer_to_runtime_config_error(tmp_path: Path) -> None:
    path = tmp_path / "overflow.yaml"
    document = valid_runtime_yaml().replace("  source_hz: 15.0", f"  source_hz: 1{'0' * 1000}")
    path.write_text(document, encoding="utf-8")

    with pytest.raises(RuntimeConfigError, match="control.source_hz must be finite"):
        load_runtime_config(path)


def test_parser_rejects_invalid_semantic_constraint(tmp_path: Path) -> None:
    runtime = copy.deepcopy(VALID_RUNTIME)
    runtime["control"]["rpc_budget_fraction"] = 1.0
    path = tmp_path / "bad.yaml"
    write_runtime(path, runtime)

    with pytest.raises(RuntimeConfigError, match="control.rpc_budget_fraction must be in \\(0, 1\\)"):
        load_runtime_config(path)


def test_parser_rejects_zero_planned_delay_steps(tmp_path: Path) -> None:
    runtime = copy.deepcopy(VALID_RUNTIME)
    runtime["rtc"]["delay"]["planned_max_steps"] = 0
    path = tmp_path / "bad.yaml"
    write_runtime(path, runtime)

    with pytest.raises(RuntimeConfigError, match="rtc.delay.planned_max_steps must be positive"):
        load_runtime_config(path)


@pytest.mark.parametrize("timeout_s", [0.0, -1.0])
def test_parser_rejects_nonpositive_initial_inference_timeout(tmp_path: Path, timeout_s: float) -> None:
    runtime = copy.deepcopy(VALID_RUNTIME)
    runtime["rtc"]["initial_inference_timeout_s"] = timeout_s
    path = tmp_path / "bad.yaml"
    write_runtime(path, runtime)

    with pytest.raises(RuntimeConfigError, match="rtc.initial_inference_timeout_s must be positive"):
        load_runtime_config(path)


@pytest.mark.parametrize("timeout_s", [0.0, -1.0])
def test_parser_rejects_nonpositive_policy_connect_timeout(tmp_path: Path, timeout_s: float) -> None:
    runtime = copy.deepcopy(VALID_RUNTIME)
    runtime["policy"]["connect_timeout_s"] = timeout_s
    path = tmp_path / "bad.yaml"
    write_runtime(path, runtime)

    with pytest.raises(RuntimeConfigError, match="policy.connect_timeout_s must be positive"):
        load_runtime_config(path)


@pytest.mark.parametrize("timeout_s", [0.0, -1.0])
def test_parser_rejects_nonpositive_command_ack_timeout(tmp_path: Path, timeout_s: float) -> None:
    runtime = copy.deepcopy(VALID_RUNTIME)
    runtime["control"]["command_ack_timeout_s"] = timeout_s
    path = tmp_path / "bad.yaml"
    write_runtime(path, runtime)

    with pytest.raises(RuntimeConfigError, match="control.command_ack_timeout_s must be positive"):
        load_runtime_config(path)


def test_parser_requires_each_policy_connect_attempt_to_finish_before_startup_timeout(tmp_path: Path) -> None:
    runtime = copy.deepcopy(VALID_RUNTIME)
    runtime["policy"]["connect_timeout_s"] = 10.0
    runtime["rtc"]["initial_inference_timeout_s"] = 10.0
    path = tmp_path / "bad.yaml"
    write_runtime(path, runtime)

    with pytest.raises(RuntimeConfigError, match="policy.connect_timeout_s must be less than rtc.initial_inference_timeout_s"):
        load_runtime_config(path)


@pytest.mark.parametrize("document", ["", "- not-a-mapping\n"])
def test_parser_rejects_empty_or_nonmapping_document(tmp_path: Path, document: str) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(document, encoding="utf-8")

    with pytest.raises(RuntimeConfigError, match="non-empty mapping"):
        load_runtime_config(path)


def test_checked_in_default_profile_loads_with_former_runtime_defaults() -> None:
    profile = Path(__file__).resolve().parents[3] / "examples" / "spirit-ai" / "configs" / "rtc" / "training_time.yaml"

    cfg = load_runtime_config(profile)

    assert (cfg.schema_version, cfg.policy.host, cfg.policy.port, cfg.policy.prompt, cfg.policy.connect_timeout_s) == (
        1,
        "localhost",
        8000,
        "fold the paper box",
        1.0,
    )
    assert (
        cfg.robot.url,
        cfg.robot.action_layout,
        cfg.robot.enable_external_following,
        cfg.robot.initial_gripper_obs_state,
        cfg.robot.gripper_initial_tolerance,
        cfg.robot.gripper_reset_command_state,
        cfg.robot.gripper_reset_steps,
    ) == ("ws://172.16.0.30:8766", "cartesian", False, 0.0965, 0.00965, 1.0, 10)
    assert (
        cfg.control.source_hz,
        cfg.control.max_steps,
        cfg.control.busy_sleep_s,
        cfg.control.startup_delay_s,
        cfg.control.blend_steps,
        cfg.control.rollback_guard_steps,
        cfg.control.rollback_scale,
        cfg.control.rpc_budget_fraction,
        cfg.control.command_ack_timeout_s,
    ) == (15.0, 2000, 0.01, 10.0, 4, 4, 0.2, 0.7, 1.0)
    assert dataclasses.asdict(cfg.control.motion_limits) == VALID_RUNTIME["control"]["motion_limits"]
    assert (cfg.rtc.mode, cfg.rtc.s_min, cfg.rtc.initial_inference_timeout_s) == ("training_time", 5, 10.0)
    assert dataclasses.asdict(cfg.rtc.delay) == {"planned_max_steps": 12, "history_window": 16, "safety_margin_steps": 1}
    assert dataclasses.asdict(cfg.rtc.deadline_miss) == {"max_consecutive": 2, "action": "hold_then_stop"}
