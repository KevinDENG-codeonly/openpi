"""Utilities for bridging SpiritAI policy outputs to robot_server joint commands."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses

import msgpack
import numpy as np

STATE_KEYS = (
    "leftarm_state_joint_pos",
    "leftarm_state_psi",
    "leftarm_gripper_state_pos",
    "rightarm_state_joint_pos",
    "rightarm_state_psi",
    "rightarm_gripper_state_pos",
    "torso_state_joint_pos",
    "base_state_speed",
)
ACTION_DIM = 27
CAMERA_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


@dataclasses.dataclass(frozen=True)
class JointMotionLimits:
    """Per-group motion limits for robot_server joint commands."""

    max_arm_velocity_rad_s: float = 0.35
    max_torso_velocity_rad_s: float = 0.2
    max_gripper_velocity_s: float = 0.8
    max_base_speed: float = 0.05
    max_joint_accel_rad_s2: float = 0.0


class RobotServerProtocolError(RuntimeError):
    """Raised when robot_server metadata or messages do not match the expected protocol."""


def map_robot_server_observation(
    obs: Mapping[str, object],
    images: Mapping[str, object],
    *,
    prompt: str,
) -> dict:
    """Maps robot_server obs/images into the SpiritAI policy observation dict."""

    missing_images = [key for key in CAMERA_KEYS if key not in images]
    if missing_images:
        raise RobotServerProtocolError(f"robot_server is missing required cameras: {missing_images}")

    missing_state = [key for key in STATE_KEYS if key not in obs]
    if missing_state:
        raise RobotServerProtocolError(f"robot_server obs is missing required state keys: {missing_state}")

    policy_obs = {key: np.asarray(images[key], dtype=np.uint8) for key in CAMERA_KEYS}
    for key in STATE_KEYS:
        policy_obs[key] = np.asarray(obs[key], dtype=np.float32).reshape(-1)
    policy_obs["prompt"] = prompt
    return policy_obs


def choose_joint_command_dim(metadata: Mapping[str, object]) -> int:
    """Chooses the widest joint command layout supported by robot_server."""

    accepted = {int(dim) for dim in metadata.get("accepted_joint_dims", [])}
    joint_dim = int(metadata.get("joint_dim", 0))

    for dim in (25, 22, 16):
        if joint_dim >= dim and (not accepted or dim in accepted):
            return dim

    raise RobotServerProtocolError(
        f"Unsupported robot_server joint metadata: joint_dim={joint_dim}, "
        f"accepted_joint_dims={sorted(accepted)}"
    )


def spiritai_actions_to_joint_commands(actions: object, target_dim: int) -> np.ndarray:
    """Converts SpiritAI 27D actions to robot_server joint command layout.

    SpiritAI action layout:
    [left_joints(7), left_psi(1), left_gripper(1),
     right_joints(7), right_psi(1), right_gripper(1),
     torso_joints(6), base_speed(3)]

    robot_server joint layouts:
    16D: [left_joints(7), left_gripper(1), right_joints(7), right_gripper(1)]
    22D: 16D + [torso_joints(6)]
    25D: 22D + [base_speed(3)]
    """

    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected actions with shape (T, 27), got {actions.shape}")
    if target_dim not in (16, 22, 25):
        raise ValueError(f"Unsupported target joint command dim: {target_dim}")

    parts = [
        actions[:, 0:7],
        actions[:, 8:9],
        actions[:, 9:16],
        actions[:, 17:18],
    ]
    if target_dim >= 22:
        parts.append(actions[:, 18:24])
    if target_dim >= 25:
        parts.append(actions[:, 24:27])

    commands = np.concatenate(parts, axis=-1)
    if commands.shape[1] != target_dim:
        raise AssertionError(f"Converted joint command has shape {commands.shape}, expected second dim {target_dim}")
    return commands.astype(np.float32, copy=False)


def robot_server_obs_to_joint_command_layout(obs: Mapping[str, object], target_dim: int) -> np.ndarray:
    """Converts robot_server state obs into the selected joint command layout."""

    if target_dim not in (16, 22, 25):
        raise ValueError(f"Unsupported target joint command dim: {target_dim}")

    parts = [
        np.asarray(obs["leftarm_state_joint_pos"], dtype=np.float32).reshape(7),
        np.asarray(obs["leftarm_gripper_state_pos"], dtype=np.float32).reshape(1),
        np.asarray(obs["rightarm_state_joint_pos"], dtype=np.float32).reshape(7),
        np.asarray(obs["rightarm_gripper_state_pos"], dtype=np.float32).reshape(1),
    ]
    if target_dim >= 22:
        parts.append(np.asarray(obs["torso_state_joint_pos"], dtype=np.float32).reshape(6))
    if target_dim >= 25:
        parts.append(np.asarray(obs["base_state_speed"], dtype=np.float32).reshape(3))

    state = np.concatenate(parts, axis=0)
    if state.shape[0] != target_dim:
        raise AssertionError(f"Converted joint state has shape {state.shape}, expected ({target_dim},)")
    return state


def joint_command_slices(target_dim: int) -> dict[str, slice]:
    """Returns named slices for a robot_server joint command vector."""

    if target_dim not in (16, 22, 25):
        raise ValueError(f"Unsupported target joint command dim: {target_dim}")

    slices = {
        "left_arm": slice(0, 7),
        "left_gripper": slice(7, 8),
        "right_arm": slice(8, 15),
        "right_gripper": slice(15, 16),
    }
    if target_dim >= 22:
        slices["torso"] = slice(16, 22)
    if target_dim >= 25:
        slices["base"] = slice(22, 25)
    return slices


def summarize_joint_delta_by_group(delta: np.ndarray, target_dim: int) -> dict[str, tuple[float, float]]:
    """Summarizes mean/max absolute delta for each robot_server joint command group."""

    delta = np.asarray(delta, dtype=np.float32).reshape(target_dim)
    summary = {}
    for name, group_slice in joint_command_slices(target_dim).items():
        group_delta = np.abs(delta[group_slice])
        summary[name] = (float(group_delta.mean()), float(group_delta.max()))
    return summary


def joint_velocity_limit_vector(target_dim: int, limits: JointMotionLimits) -> np.ndarray:
    """Returns per-dimension velocity limits for joint-position command dims."""

    if target_dim not in (16, 22, 25):
        raise ValueError(f"Unsupported target joint command dim: {target_dim}")
    limit = np.full(target_dim, np.inf, dtype=np.float32)
    slices = joint_command_slices(target_dim)
    limit[slices["left_arm"]] = limits.max_arm_velocity_rad_s
    limit[slices["left_gripper"]] = limits.max_gripper_velocity_s
    limit[slices["right_arm"]] = limits.max_arm_velocity_rad_s
    limit[slices["right_gripper"]] = limits.max_gripper_velocity_s
    if target_dim >= 22:
        limit[slices["torso"]] = limits.max_torso_velocity_rad_s
    if target_dim >= 25:
        # Base command dims are already speed commands, not positions.
        limit[slices["base"]] = np.inf
    return limit


def limit_joint_command_motion(
    commands: object,
    current_state: object,
    *,
    source_hz: float,
    limits: JointMotionLimits,
    previous_commands: object | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Limits joint command velocity, with optional joint acceleration limiting.

    Arm, gripper, and torso dimensions are absolute command targets and are
    limited by adjacent-frame velocity. Base dimensions are speed commands, so
    they are clamped by absolute speed instead.
    """

    commands = np.asarray(commands, dtype=np.float32)
    current_state = np.asarray(current_state, dtype=np.float32).reshape(-1)
    if commands.ndim != 2:
        raise ValueError(f"Expected commands with shape (T, D), got {commands.shape}")
    if current_state.shape != (commands.shape[1],):
        raise ValueError(f"Expected current_state shape ({commands.shape[1]},), got {current_state.shape}")
    if source_hz <= 0:
        raise ValueError(f"source_hz must be positive, got {source_hz}")

    target_dim = commands.shape[1]
    velocity_limit = joint_velocity_limit_vector(target_dim, limits)
    step_limit = velocity_limit / float(source_hz)
    out = np.empty_like(commands)
    prev = current_state.copy()

    limited_values = 0
    total_limited_dims = 0
    max_raw_velocity = 0.0
    max_limited_velocity = 0.0
    prev_velocity = np.zeros(target_dim, dtype=np.float32)
    if (
        previous_commands is not None
        and limits.max_joint_accel_rad_s2 > 0.0
        and np.asarray(previous_commands).shape == commands.shape
        and commands.shape[0] >= 2
    ):
        previous_commands = np.asarray(previous_commands, dtype=np.float32)
        prev_velocity = (previous_commands[-1] - previous_commands[-2]) * float(source_hz)

    accel_step = limits.max_joint_accel_rad_s2 / float(source_hz)
    accel_limit = np.full(target_dim, accel_step, dtype=np.float32)
    if target_dim >= 25:
        accel_limit[joint_command_slices(target_dim)["base"]] = np.inf

    for i, command in enumerate(commands):
        raw_delta = command - prev
        raw_velocity = raw_delta * float(source_hz)
        finite_velocity = raw_velocity[np.isfinite(velocity_limit)]
        if finite_velocity.size:
            max_raw_velocity = max(max_raw_velocity, float(np.max(np.abs(finite_velocity))))

        limited_delta = np.clip(raw_delta, -step_limit, step_limit)
        if limits.max_joint_accel_rad_s2 > 0.0:
            limited_velocity = limited_delta * float(source_hz)
            limited_velocity = np.clip(limited_velocity, prev_velocity - accel_limit, prev_velocity + accel_limit)
            limited_delta = np.clip(limited_velocity / float(source_hz), -step_limit, step_limit)
            prev_velocity = limited_delta * float(source_hz)

        changed = np.abs(limited_delta - raw_delta) > 1e-6
        limited_values += int(np.count_nonzero(changed & np.isfinite(step_limit)))
        total_limited_dims += int(np.count_nonzero(np.isfinite(step_limit)))
        next_command = prev + limited_delta

        if target_dim >= 25 and np.isfinite(limits.max_base_speed):
            base_slice = joint_command_slices(target_dim)["base"]
            raw_base = next_command[base_slice].copy()
            next_command[base_slice] = np.clip(raw_base, -limits.max_base_speed, limits.max_base_speed)
            limited_values += int(np.count_nonzero(np.abs(next_command[base_slice] - raw_base) > 1e-6))
            total_limited_dims += next_command[base_slice].size

        velocity = (next_command - prev) * float(source_hz)
        finite_velocity = velocity[np.isfinite(velocity_limit)]
        if finite_velocity.size:
            max_limited_velocity = max(max_limited_velocity, float(np.max(np.abs(finite_velocity))))
        out[i] = next_command
        prev = next_command

    stats = {
        "max_raw_velocity": max_raw_velocity,
        "max_limited_velocity": max_limited_velocity,
        "limited_fraction": float(limited_values / total_limited_dims) if total_limited_dims else 0.0,
    }
    return out, stats


def blend_joint_command_start(commands: object, current_state: object, blend_steps: int) -> np.ndarray:
    """Aligns the first command to current_state and blends the first N commands.

    The first command is exactly the current robot state. The following commands
    linearly transition toward the raw policy commands over ``blend_steps``.
    """

    commands = np.asarray(commands, dtype=np.float32).copy()
    current_state = np.asarray(current_state, dtype=np.float32).reshape(-1)
    if commands.ndim != 2:
        raise ValueError(f"Expected commands with shape (T, D), got {commands.shape}")
    if current_state.shape != (commands.shape[1],):
        raise ValueError(f"Expected current_state shape ({commands.shape[1]},), got {current_state.shape}")
    if blend_steps <= 0 or commands.shape[0] == 0:
        return commands

    steps = min(blend_steps, commands.shape[0])
    raw = commands[:steps].copy()
    alphas = np.linspace(0.0, 1.0, steps, dtype=np.float32)[:, None]
    commands[:steps] = (1.0 - alphas) * current_state[None, :] + alphas * raw
    return commands


def suppress_chunk_start_rollback(
    commands: object,
    previous_commands: object | None,
    *,
    guard_steps: int,
    rollback_scale: float,
) -> np.ndarray:
    """Suppresses per-dimension rollback at the start of a new command chunk.

    A rollback is a new-chunk motion component that points opposite the previous
    chunk's tail velocity. Only arm and torso dimensions are guarded; grippers
    and base are left untouched.
    """

    commands = np.asarray(commands, dtype=np.float32).copy()
    if previous_commands is None or guard_steps <= 0 or commands.shape[0] == 0:
        return commands
    previous_commands = np.asarray(previous_commands, dtype=np.float32)
    if commands.ndim != 2 or previous_commands.ndim != 2:
        raise ValueError(f"Expected 2D command chunks, got {commands.shape} and {previous_commands.shape}")
    if commands.shape[1] != previous_commands.shape[1]:
        raise ValueError(f"Command dims do not match: {commands.shape[1]} vs {previous_commands.shape[1]}")
    if previous_commands.shape[0] < 2:
        return commands
    if not 0.0 <= rollback_scale <= 1.0:
        raise ValueError(f"rollback_scale must be in [0, 1], got {rollback_scale}")

    guarded_slices = [joint_command_slices(commands.shape[1])[name] for name in ("left_arm", "right_arm")]
    if commands.shape[1] >= 22:
        guarded_slices.append(joint_command_slices(commands.shape[1])["torso"])

    previous_tail = previous_commands[-1]
    previous_velocity = previous_commands[-1] - previous_commands[-2]
    steps = min(guard_steps, commands.shape[0])
    for group_slice in guarded_slices:
        raw_delta = commands[:steps, group_slice] - previous_tail[group_slice]
        tail_velocity = previous_velocity[group_slice][None, :]
        rollback_mask = raw_delta * tail_velocity < 0.0
        commands[:steps, group_slice] = np.where(
            rollback_mask,
            previous_tail[group_slice][None, :] + raw_delta * rollback_scale,
            commands[:steps, group_slice],
        )
    return commands


def _pack_numpy_for_robot_server(obj: object) -> object:
    """Encodes NumPy values in the same wire format as the msgpack-numpy package."""

    if isinstance(obj, np.ndarray):
        return {
            b"nd": True,
            b"type": obj.dtype.str,
            b"kind": b"",
            b"shape": obj.shape,
            b"data": obj.tobytes(),
        }
    if isinstance(obj, np.generic):
        value = np.asarray(obj)
        return {
            b"nd": False,
            b"type": value.dtype.str,
            b"data": value.tobytes(),
        }
    return obj


def _unpack_numpy_from_robot_server(obj: dict) -> object:
    if b"nd" not in obj:
        return obj
    dtype = np.dtype(obj[b"type"])
    if obj[b"nd"] is True:
        return np.ndarray(buffer=obj[b"data"], dtype=dtype, shape=obj[b"shape"])
    return np.frombuffer(obj[b"data"], dtype=dtype)[0]


def pack_robot_server_message(obj: Mapping[str, object]) -> bytes:
    return msgpack.packb(obj, default=_pack_numpy_for_robot_server, use_bin_type=True)


def unpack_robot_server_message(buf: bytes | str) -> dict:
    data = buf if isinstance(buf, bytes) else buf.encode()
    return msgpack.unpackb(data, object_hook=_unpack_numpy_from_robot_server, raw=False)
