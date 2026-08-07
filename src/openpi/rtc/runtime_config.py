"""Strict YAML configuration for the training-time RTC runtime."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
import dataclasses
import math
import numbers
from pathlib import Path
from typing import Any

import yaml


class RuntimeConfigError(ValueError):
    """Raised when a runtime configuration is invalid."""


class _StrictSafeLoader(yaml.SafeLoader):
    """SafeLoader variant that rejects duplicate or invalid mapping keys."""

    def construct_mapping(
        self, node: yaml.nodes.MappingNode, deep: bool = False  # noqa: FBT001, FBT002
    ) -> dict[Any, Any]:
        if not isinstance(node, yaml.nodes.MappingNode):
            return super().construct_mapping(node, deep=deep)

        mapping = {}
        for key_node, value_node in node.value:
            try:
                key = self.construct_object(key_node, deep=deep)
            except yaml.YAMLError as exc:
                raise RuntimeConfigError("invalid mapping key") from exc
            if not isinstance(key, Hashable):
                raise RuntimeConfigError(f"unhashable mapping key {key!r}")
            if key in mapping:
                raise RuntimeConfigError(f"duplicate mapping key {key!r}")
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


@dataclasses.dataclass(frozen=True)
class PolicyRuntimeConfig:
    host: str
    port: int
    prompt: str


@dataclasses.dataclass(frozen=True)
class RobotRuntimeConfig:
    url: str
    action_layout: str
    enable_external_following: bool
    initial_gripper_obs_state: float
    gripper_initial_tolerance: float
    gripper_reset_command_state: float
    gripper_reset_steps: int


@dataclasses.dataclass(frozen=True)
class MotionLimitsRuntimeConfig:
    max_arm_velocity_rad_s: float
    max_torso_velocity_rad_s: float
    max_gripper_velocity_s: float
    max_base_speed: float
    max_joint_accel_rad_s2: float
    max_cart_translation_m_s: float
    max_cart_rotation_rad_s: float
    max_torso_cart_translation_m_s: float
    max_torso_cart_rotation_rad_s: float
    max_cart_accel: float


@dataclasses.dataclass(frozen=True)
class ControlRuntimeConfig:
    source_hz: float
    max_steps: int
    busy_sleep_s: float
    startup_delay_s: float
    blend_steps: int
    rollback_guard_steps: int
    rollback_scale: float
    rpc_budget_fraction: float
    motion_limits: MotionLimitsRuntimeConfig


@dataclasses.dataclass(frozen=True)
class RTCDelayRuntimeConfig:
    planned_max_steps: int
    history_window: int
    safety_margin_steps: int


@dataclasses.dataclass(frozen=True)
class RTCDeadlineRuntimeConfig:
    max_consecutive: int
    action: str


@dataclasses.dataclass(frozen=True)
class RTCRuntimeConfig:
    mode: str
    s_min: int
    delay: RTCDelayRuntimeConfig
    deadline_miss: RTCDeadlineRuntimeConfig


@dataclasses.dataclass(frozen=True)
class RuntimeConfig:
    schema_version: int
    policy: PolicyRuntimeConfig
    robot: RobotRuntimeConfig
    control: ControlRuntimeConfig
    rtc: RTCRuntimeConfig


def default_config_path(entrypoint: Path) -> Path:
    """Return the default profile path without reading from the filesystem."""
    return Path(entrypoint).parent / "configs" / "rtc" / "training_time.yaml"


def load_runtime_config(path: Path) -> RuntimeConfig:
    """Load a complete training-time RTC runtime profile from YAML."""
    config_path = Path(path)
    try:
        document = yaml.load(config_path.read_text(encoding="utf-8"), Loader=_StrictSafeLoader)
    except UnicodeDecodeError as exc:
        raise RuntimeConfigError(f"Runtime config {config_path} must be valid UTF-8") from exc
    except OSError as exc:
        raise RuntimeConfigError(f"Unable to read runtime config {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise RuntimeConfigError(f"Invalid YAML in runtime config {config_path}: {exc}") from exc

    root = _require_nonempty_mapping(document, "runtime config")
    _require_exact_keys(root, {"schema_version", "policy", "robot", "control", "rtc"}, "runtime config")

    schema_version = _require_integer(root["schema_version"], "schema_version")
    if schema_version != 1:
        raise RuntimeConfigError(f"schema_version must be 1, got {schema_version}")

    return RuntimeConfig(
        schema_version=schema_version,
        policy=_parse_policy(root["policy"]),
        robot=_parse_robot(root["robot"]),
        control=_parse_control(root["control"]),
        rtc=_parse_rtc(root["rtc"]),
    )


def _parse_policy(value: Any) -> PolicyRuntimeConfig:
    mapping = _require_mapping(value, "policy")
    _require_exact_keys(mapping, {"host", "port", "prompt"}, "policy")
    host = _require_nonempty_string(mapping["host"], "policy.host")
    port = _require_positive_integer(mapping["port"], "policy.port")
    if port > 65535:
        raise RuntimeConfigError(f"policy.port must be at most 65535, got {port}")
    return PolicyRuntimeConfig(host=host, port=port, prompt=_require_nonempty_string(mapping["prompt"], "policy.prompt"))


def _parse_robot(value: Any) -> RobotRuntimeConfig:
    mapping = _require_mapping(value, "robot")
    _require_exact_keys(
        mapping,
        {
            "url",
            "action_layout",
            "enable_external_following",
            "initial_gripper_obs_state",
            "gripper_initial_tolerance",
            "gripper_reset_command_state",
            "gripper_reset_steps",
        },
        "robot",
    )
    url = _require_nonempty_string(mapping["url"], "robot.url")
    if not url.startswith(("ws://", "wss://")):
        raise RuntimeConfigError("robot.url must use ws:// or wss://")
    action_layout = _require_nonempty_string(mapping["action_layout"], "robot.action_layout")
    if action_layout not in {"joint", "cartesian"}:
        raise RuntimeConfigError("robot.action_layout must be 'joint' or 'cartesian'")
    return RobotRuntimeConfig(
        url=url,
        action_layout=action_layout,
        enable_external_following=_require_boolean(mapping["enable_external_following"], "robot.enable_external_following"),
        initial_gripper_obs_state=_require_finite_real(
            mapping["initial_gripper_obs_state"], "robot.initial_gripper_obs_state"
        ),
        gripper_initial_tolerance=_require_nonnegative_real(
            mapping["gripper_initial_tolerance"], "robot.gripper_initial_tolerance"
        ),
        gripper_reset_command_state=_require_finite_real(
            mapping["gripper_reset_command_state"], "robot.gripper_reset_command_state"
        ),
        gripper_reset_steps=_require_positive_integer(mapping["gripper_reset_steps"], "robot.gripper_reset_steps"),
    )


def _parse_control(value: Any) -> ControlRuntimeConfig:
    mapping = _require_mapping(value, "control")
    _require_exact_keys(
        mapping,
        {
            "source_hz",
            "max_steps",
            "busy_sleep_s",
            "startup_delay_s",
            "blend_steps",
            "rollback_guard_steps",
            "rollback_scale",
            "rpc_budget_fraction",
            "motion_limits",
        },
        "control",
    )
    source_hz = _require_positive_real(mapping["source_hz"], "control.source_hz")
    rpc_budget_fraction = _require_finite_real(mapping["rpc_budget_fraction"], "control.rpc_budget_fraction")
    if not 0 < rpc_budget_fraction < 1:
        raise RuntimeConfigError("control.rpc_budget_fraction must be in (0, 1)")
    rollback_scale = _require_nonnegative_real(mapping["rollback_scale"], "control.rollback_scale")
    if rollback_scale > 1:
        raise RuntimeConfigError("control.rollback_scale must be at most 1")
    return ControlRuntimeConfig(
        source_hz=source_hz,
        max_steps=_require_positive_integer(mapping["max_steps"], "control.max_steps"),
        busy_sleep_s=_require_nonnegative_real(mapping["busy_sleep_s"], "control.busy_sleep_s"),
        startup_delay_s=_require_nonnegative_real(mapping["startup_delay_s"], "control.startup_delay_s"),
        blend_steps=_require_nonnegative_integer(mapping["blend_steps"], "control.blend_steps"),
        rollback_guard_steps=_require_nonnegative_integer(
            mapping["rollback_guard_steps"], "control.rollback_guard_steps"
        ),
        rollback_scale=rollback_scale,
        rpc_budget_fraction=rpc_budget_fraction,
        motion_limits=_parse_motion_limits(mapping["motion_limits"]),
    )


def _parse_motion_limits(value: Any) -> MotionLimitsRuntimeConfig:
    mapping = _require_mapping(value, "control.motion_limits")
    _require_exact_keys(
        mapping,
        {
            "max_arm_velocity_rad_s",
            "max_torso_velocity_rad_s",
            "max_gripper_velocity_s",
            "max_base_speed",
            "max_joint_accel_rad_s2",
            "max_cart_translation_m_s",
            "max_cart_rotation_rad_s",
            "max_torso_cart_translation_m_s",
            "max_torso_cart_rotation_rad_s",
            "max_cart_accel",
        },
        "control.motion_limits",
    )
    return MotionLimitsRuntimeConfig(
        max_arm_velocity_rad_s=_require_nonnegative_real(
            mapping["max_arm_velocity_rad_s"], "control.motion_limits.max_arm_velocity_rad_s"
        ),
        max_torso_velocity_rad_s=_require_nonnegative_real(
            mapping["max_torso_velocity_rad_s"], "control.motion_limits.max_torso_velocity_rad_s"
        ),
        max_gripper_velocity_s=_require_nonnegative_real(
            mapping["max_gripper_velocity_s"], "control.motion_limits.max_gripper_velocity_s"
        ),
        max_base_speed=_require_nonnegative_real(mapping["max_base_speed"], "control.motion_limits.max_base_speed"),
        max_joint_accel_rad_s2=_require_nonnegative_real(
            mapping["max_joint_accel_rad_s2"], "control.motion_limits.max_joint_accel_rad_s2"
        ),
        max_cart_translation_m_s=_require_nonnegative_real(
            mapping["max_cart_translation_m_s"], "control.motion_limits.max_cart_translation_m_s"
        ),
        max_cart_rotation_rad_s=_require_nonnegative_real(
            mapping["max_cart_rotation_rad_s"], "control.motion_limits.max_cart_rotation_rad_s"
        ),
        max_torso_cart_translation_m_s=_require_nonnegative_real(
            mapping["max_torso_cart_translation_m_s"], "control.motion_limits.max_torso_cart_translation_m_s"
        ),
        max_torso_cart_rotation_rad_s=_require_nonnegative_real(
            mapping["max_torso_cart_rotation_rad_s"], "control.motion_limits.max_torso_cart_rotation_rad_s"
        ),
        max_cart_accel=_require_nonnegative_real(mapping["max_cart_accel"], "control.motion_limits.max_cart_accel"),
    )


def _parse_rtc(value: Any) -> RTCRuntimeConfig:
    mapping = _require_mapping(value, "rtc")
    _require_exact_keys(mapping, {"mode", "s_min", "delay", "deadline_miss"}, "rtc")
    mode = _require_nonempty_string(mapping["mode"], "rtc.mode")
    if mode != "training_time":
        raise RuntimeConfigError("rtc.mode must be 'training_time'")
    return RTCRuntimeConfig(
        mode=mode,
        s_min=_require_nonnegative_integer(mapping["s_min"], "rtc.s_min"),
        delay=_parse_delay(mapping["delay"]),
        deadline_miss=_parse_deadline_miss(mapping["deadline_miss"]),
    )


def _parse_delay(value: Any) -> RTCDelayRuntimeConfig:
    mapping = _require_mapping(value, "rtc.delay")
    _require_exact_keys(mapping, {"planned_max_steps", "history_window", "safety_margin_steps"}, "rtc.delay")
    return RTCDelayRuntimeConfig(
        planned_max_steps=_require_positive_integer(mapping["planned_max_steps"], "rtc.delay.planned_max_steps"),
        history_window=_require_nonnegative_integer(mapping["history_window"], "rtc.delay.history_window"),
        safety_margin_steps=_require_nonnegative_integer(
            mapping["safety_margin_steps"], "rtc.delay.safety_margin_steps"
        ),
    )


def _parse_deadline_miss(value: Any) -> RTCDeadlineRuntimeConfig:
    mapping = _require_mapping(value, "rtc.deadline_miss")
    _require_exact_keys(mapping, {"max_consecutive", "action"}, "rtc.deadline_miss")
    action = _require_nonempty_string(mapping["action"], "rtc.deadline_miss.action")
    if action != "hold_then_stop":
        raise RuntimeConfigError("rtc.deadline_miss.action must be 'hold_then_stop'")
    return RTCDeadlineRuntimeConfig(
        max_consecutive=_require_nonnegative_integer(mapping["max_consecutive"], "rtc.deadline_miss.max_consecutive"),
        action=action,
    )


def _require_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeConfigError(f"{location} must be a mapping")
    return value


def _require_nonempty_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise RuntimeConfigError(f"{location} must be a non-empty mapping")
    return value


def _require_exact_keys(mapping: Mapping[str, Any], expected_keys: set[str], location: str) -> None:
    actual_keys = set(mapping)
    unknown_keys = actual_keys - expected_keys
    missing_keys = expected_keys - actual_keys
    if unknown_keys or missing_keys:
        messages = []
        if unknown_keys:
            messages.append(f"unknown keys {_format_keys(unknown_keys)}")
        if missing_keys:
            messages.append(f"missing keys {_format_keys(missing_keys)}")
        raise RuntimeConfigError(f"{location}: {', '.join(messages)}")


def _format_keys(keys: set[object]) -> str:
    return f"[{', '.join(sorted(repr(key) for key in keys))}]"


def _require_nonempty_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeConfigError(f"{location} must be a non-empty string")
    return value


def _require_boolean(value: Any, location: str) -> bool:
    if not isinstance(value, bool):
        raise RuntimeConfigError(f"{location} must be a boolean")
    return value


def _require_integer(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise RuntimeConfigError(f"{location} must be an integer")
    return int(value)


def _require_nonnegative_integer(value: Any, location: str) -> int:
    integer = _require_integer(value, location)
    if integer < 0:
        raise RuntimeConfigError(f"{location} must be non-negative")
    return integer


def _require_positive_integer(value: Any, location: str) -> int:
    integer = _require_integer(value, location)
    if integer <= 0:
        raise RuntimeConfigError(f"{location} must be positive")
    return integer


def _require_finite_real(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise RuntimeConfigError(f"{location} must be a number")
    try:
        real = float(value)
    except (OverflowError, ValueError) as exc:
        raise RuntimeConfigError(f"{location} must be finite") from exc
    if not math.isfinite(real):
        raise RuntimeConfigError(f"{location} must be finite")
    return real


def _require_nonnegative_real(value: Any, location: str) -> float:
    real = _require_finite_real(value, location)
    if real < 0:
        raise RuntimeConfigError(f"{location} must be non-negative")
    return real


def _require_positive_real(value: Any, location: str) -> float:
    real = _require_finite_real(value, location)
    if real <= 0:
        raise RuntimeConfigError(f"{location} must be positive")
    return real
