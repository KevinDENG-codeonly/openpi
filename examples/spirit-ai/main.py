import dataclasses
import logging
import time
from typing import Literal

import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro
import websockets.sync.client

from openpi.policies import spiritai_bridge


@dataclasses.dataclass
class Args:
    policy_host: str = "localhost"
    policy_port: int = 8000
    robot_url: str = "ws://172.16.0.30:8766"
    prompt: str = "fold the paper box"
    max_steps: int = 2000
    policy_action_layout: Literal["joint", "cartesian"] = "joint"
    execute_steps: int | None = None
    source_hz: float = 15.0
    busy_sleep_s: float = 0.01
    startup_delay_s: float = 10.0
    enable_external_following: bool = False
    blend_steps: int = 4
    rollback_guard_steps: int = 4
    rollback_scale: float = 0.2
    prefetch_next_chunk: bool = True
    prefetch_delay_fraction: float = 0.5
    max_arm_velocity_rad_s: float = 0.35
    max_torso_velocity_rad_s: float = 0.2
    max_gripper_velocity_s: float = 0.8
    max_base_speed: float = 0.05
    max_joint_accel_rad_s2: float = 0.0
    max_cart_translation_m_s: float = 0.08
    max_cart_rotation_rad_s: float = 0.35
    max_torso_cart_translation_m_s: float = 0.04
    max_torso_cart_rotation_rad_s: float = 0.2
    max_cart_accel: float = 0.0
    initial_gripper_obs_state: float = 0.0965
    gripper_initial_tolerance: float = 0.00965
    gripper_reset_command_state: float = 1.0
    gripper_reset_steps: int = 10


def _wait_until_robot_idle(robot_ws: websockets.sync.client.ClientConnection, busy_sleep_s: float) -> None:
    while True:
        robot_ws.send(spiritai_bridge.pack_robot_server_message({"type": "get_status"}))
        status = spiritai_bridge.unpack_robot_server_message(robot_ws.recv())
        if not status["busy"]:
            return
        time.sleep(busy_sleep_s)


def _get_robot_obs(robot_ws: websockets.sync.client.ClientConnection) -> tuple[dict, dict]:
    robot_ws.send(spiritai_bridge.pack_robot_server_message({"type": "get_obs"}))
    msg = spiritai_bridge.unpack_robot_server_message(robot_ws.recv())
    if msg.get("type") not in (None, "obs"):
        raise spiritai_bridge.RobotServerProtocolError(f"Unexpected get_obs response: {msg.get('type')}")
    return msg["obs"], msg["images"]


def _set_robot_external_following(robot_ws: websockets.sync.client.ClientConnection, *, enabled: bool) -> None:
    robot_ws.send(spiritai_bridge.pack_robot_server_message({"type": "set_external_following", "enabled": enabled}))
    msg = spiritai_bridge.unpack_robot_server_message(robot_ws.recv())
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


def _get_scalar_obs_value(obs: dict, key: str) -> float:
    value = np.asarray(obs[key], dtype=np.float32).reshape(-1)
    if value.size != 1:
        raise spiritai_bridge.RobotServerProtocolError(f"Expected scalar obs value for {key}, got shape {value.shape}")
    return float(value[0])


def _get_gripper_state(obs: dict) -> tuple[float, float]:
    return (
        _get_scalar_obs_value(obs, "leftarm_gripper_state_pos"),
        _get_scalar_obs_value(obs, "rightarm_gripper_state_pos"),
    )


def _grippers_at_initial_state(
    obs: dict,
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
    obs: dict,
    *,
    policy_action_layout: Literal["joint", "cartesian"],
    robot_command_kind: str,
    command_dim: int,
    gripper_reset_command_state: float,
    gripper_reset_steps: int,
    source_hz: float,
) -> None:
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
    ack = spiritai_bridge.unpack_robot_server_message(robot_ws.recv())
    if not ack.get("accepted", False):
        raise spiritai_bridge.RobotServerProtocolError(f"Initial gripper reset rejected: {ack.get('error')}")
    logging.info(
        "Initial gripper reset accepted: chunk_id=%s actions=%s expected_finish_at=%s",
        ack.get("chunk_id"),
        reset_commands.shape,
        ack.get("expected_finish_at"),
    )


def _infer_policy_actions(
    policy: _websocket_client_policy.WebsocketClientPolicy,
    obs: dict,
    images: dict,
    *,
    prompt: str,
    policy_action_layout: Literal["joint", "cartesian"],
) -> object:
    if policy_action_layout == "cartesian":
        policy_obs = spiritai_bridge.map_robot_server_cartesian_observation(obs, images, prompt=prompt)
    else:
        policy_obs = spiritai_bridge.map_robot_server_observation(obs, images, prompt=prompt)
    return policy.infer(policy_obs)["actions"]


def main(args: Args) -> None:
    if not 0.0 <= args.prefetch_delay_fraction <= 1.0:
        raise ValueError(f"prefetch_delay_fraction must be in [0, 1], got {args.prefetch_delay_fraction}")
    if args.execute_steps is not None and args.execute_steps <= 0:
        raise ValueError(f"execute_steps must be positive when set, got {args.execute_steps}")
    if args.gripper_initial_tolerance < 0:
        raise ValueError(f"gripper_initial_tolerance must be non-negative, got {args.gripper_initial_tolerance}")
    if args.gripper_reset_steps <= 0:
        raise ValueError(f"gripper_reset_steps must be positive, got {args.gripper_reset_steps}")

    policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.policy_host,
        port=args.policy_port,
    )
    logging.info("Policy server metadata: %s", policy.get_server_metadata())

    with websockets.sync.client.connect(args.robot_url, max_size=None, compression=None) as robot_ws:
        hello = spiritai_bridge.unpack_robot_server_message(robot_ws.recv())
        if hello.get("type") != "hello":
            raise spiritai_bridge.RobotServerProtocolError(
                f"Expected hello from robot_server, got: {hello.get('type')}"
            )

        metadata = hello["metadata"]
        if args.policy_action_layout == "cartesian":
            command_dim = spiritai_bridge.choose_cartesian_command_dim(metadata)
            robot_command_kind = "cart"
        else:
            command_dim = spiritai_bridge.choose_joint_command_dim(metadata)
            robot_command_kind = "joint"
        logging.info(
            "Robot server connected: structure=%s layout=%s command_kind=%s command_dim=%s "
            "cart_dim=%s joint_dim=%s cameras=%s",
            metadata.get("structure"),
            args.policy_action_layout,
            robot_command_kind,
            command_dim,
            metadata.get("cart_dim"),
            metadata.get("joint_dim"),
            metadata.get("cameras"),
        )
        if args.enable_external_following:
            _set_robot_external_following(robot_ws, enabled=True)
        if args.startup_delay_s > 0:
            logging.info("Startup delay %.1fs before first inference.", args.startup_delay_s)
            time.sleep(args.startup_delay_s)

        max_chunk = int(metadata.get("max_chunk", 60))
        if args.gripper_reset_steps > max_chunk:
            raise ValueError(
                f"gripper_reset_steps must be <= robot_server max_chunk ({max_chunk}), got {args.gripper_reset_steps}"
            )
        motion_limits = spiritai_bridge.JointMotionLimits(
            max_arm_velocity_rad_s=args.max_arm_velocity_rad_s,
            max_torso_velocity_rad_s=args.max_torso_velocity_rad_s,
            max_gripper_velocity_s=args.max_gripper_velocity_s,
            max_base_speed=args.max_base_speed,
            max_joint_accel_rad_s2=args.max_joint_accel_rad_s2,
        )
        cartesian_motion_limits = spiritai_bridge.CartesianMotionLimits(
            max_arm_translation_m_s=args.max_cart_translation_m_s,
            max_arm_rotation_rad_s=args.max_cart_rotation_rad_s,
            max_torso_translation_m_s=args.max_torso_cart_translation_m_s,
            max_torso_rotation_rad_s=args.max_torso_cart_rotation_rad_s,
            max_gripper_velocity_s=args.max_gripper_velocity_s,
            max_base_speed=args.max_base_speed,
            max_cart_accel=args.max_cart_accel,
        )

        _wait_until_robot_idle(robot_ws, args.busy_sleep_s)
        initial_obs, _ = _get_robot_obs(robot_ws)
        left_gripper, right_gripper = _get_gripper_state(initial_obs)
        if _grippers_at_initial_state(
            initial_obs,
            initial_gripper_obs_state=args.initial_gripper_obs_state,
            tolerance=args.gripper_initial_tolerance,
        ):
            logging.info(
                "Initial gripper state OK: left=%.4f right=%.4f target=%.4f",
                left_gripper,
                right_gripper,
                args.initial_gripper_obs_state,
            )
        else:
            logging.warning(
                "Initial gripper state is not reset: left=%.4f right=%.4f target=%.4f tolerance=%.4f; "
                "resetting before inference with command=%.4f.",
                left_gripper,
                right_gripper,
                args.initial_gripper_obs_state,
                args.gripper_initial_tolerance,
                args.gripper_reset_command_state,
            )
            _send_initial_gripper_reset(
                robot_ws,
                initial_obs,
                policy_action_layout=args.policy_action_layout,
                robot_command_kind=robot_command_kind,
                command_dim=command_dim,
                gripper_reset_command_state=args.gripper_reset_command_state,
                gripper_reset_steps=args.gripper_reset_steps,
                source_hz=args.source_hz,
            )
            _wait_until_robot_idle(robot_ws, args.busy_sleep_s)
            verify_obs, _ = _get_robot_obs(robot_ws)
            verify_left_gripper, verify_right_gripper = _get_gripper_state(verify_obs)
            if not _grippers_at_initial_state(
                verify_obs,
                initial_gripper_obs_state=args.initial_gripper_obs_state,
                tolerance=args.gripper_initial_tolerance,
            ):
                raise RuntimeError(
                    "Initial gripper reset did not reach target: "
                    f"left={verify_left_gripper:.4f} right={verify_right_gripper:.4f} "
                    f"target={args.initial_gripper_obs_state:.4f} "
                    f"tolerance={args.gripper_initial_tolerance:.4f}"
                )
            logging.info(
                "Initial gripper reset verified: left=%.4f right=%.4f target=%.4f",
                verify_left_gripper,
                verify_right_gripper,
                args.initial_gripper_obs_state,
            )

        previous_state = None
        previous_commands = None
        prefetched_actions = None
        for step in range(args.max_steps):
            _wait_until_robot_idle(robot_ws, args.busy_sleep_s)
            obs, images = _get_robot_obs(robot_ws)
            if args.policy_action_layout == "cartesian":
                current_state = spiritai_bridge.robot_server_obs_to_cartesian_command_layout(obs, command_dim)
                summarize_delta = spiritai_bridge.summarize_cartesian_delta_by_group
            else:
                current_state = spiritai_bridge.robot_server_obs_to_joint_command_layout(obs, command_dim)
                summarize_delta = spiritai_bridge.summarize_joint_delta_by_group
            if previous_state is not None:
                actual_delta = current_state - previous_state
                actual_delta_summary = summarize_delta(actual_delta, command_dim)
                logging.info(
                    "Step %d actual state delta since previous chunk: %s",
                    step,
                    " ".join(
                        f"{name}=mean:{mean_abs:.4f}/max:{max_abs:.4f}"
                        for name, (mean_abs, max_abs) in actual_delta_summary.items()
                    ),
                )

            if prefetched_actions is None:
                inference_start_s = time.perf_counter()
                raw_actions = _infer_policy_actions(
                    policy,
                    obs,
                    images,
                    prompt=args.prompt,
                    policy_action_layout=args.policy_action_layout,
                )
                inference_latency_s = time.perf_counter() - inference_start_s
                logging.info("Step %d policy inference latency: %.3fs", step, inference_latency_s)
            else:
                raw_actions = prefetched_actions
                prefetched_actions = None
                logging.info("Step %d using prefetched policy actions", step)

            if args.policy_action_layout == "cartesian":
                commands = spiritai_bridge.spiritai_cartesian_actions_to_cartesian_commands(raw_actions, command_dim)
            else:
                commands = spiritai_bridge.spiritai_actions_to_joint_commands(raw_actions, command_dim)
            if args.execute_steps is not None:
                commands = commands[: args.execute_steps]
            if commands.shape[0] > max_chunk:
                commands = commands[:max_chunk]
            if args.policy_action_layout == "cartesian":
                commands = spiritai_bridge.suppress_cartesian_chunk_start_rollback(
                    commands,
                    previous_commands,
                    guard_steps=args.rollback_guard_steps,
                    rollback_scale=args.rollback_scale,
                )
            else:
                commands = spiritai_bridge.suppress_chunk_start_rollback(
                    commands,
                    previous_commands,
                    guard_steps=args.rollback_guard_steps,
                    rollback_scale=args.rollback_scale,
                )
            commands = spiritai_bridge.blend_joint_command_start(
                commands,
                current_state,
                args.blend_steps,
            )
            if args.policy_action_layout == "cartesian":
                commands, motion_stats = spiritai_bridge.limit_cartesian_command_motion(
                    commands,
                    current_state,
                    source_hz=args.source_hz,
                    limits=cartesian_motion_limits,
                    previous_commands=previous_commands,
                )
            else:
                commands, motion_stats = spiritai_bridge.limit_joint_command_motion(
                    commands,
                    current_state,
                    source_hz=args.source_hz,
                    limits=motion_limits,
                    previous_commands=previous_commands,
                )
            first_delta = commands[0] - current_state
            delta_summary = summarize_delta(first_delta, command_dim)
            logging.info(
                "Step %d command stats: min=%.4f max=%.4f first_delta_mean_abs=%.4f first_delta_max_abs=%.4f",
                step,
                float(commands.min()),
                float(commands.max()),
                float(abs(first_delta).mean()),
                float(abs(first_delta).max()),
            )
            logging.info(
                "Step %d motion limit: raw_max_vel=%.4f limited_max_vel=%.4f limited_fraction=%.2f",
                step,
                motion_stats["max_raw_velocity"],
                motion_stats["max_limited_velocity"],
                motion_stats["limited_fraction"],
            )
            logging.info(
                "Step %d delta by group: %s",
                step,
                " ".join(
                    f"{name}=mean:{mean_abs:.4f}/max:{max_abs:.4f}"
                    for name, (mean_abs, max_abs) in delta_summary.items()
                ),
            )

            robot_ws.send(
                spiritai_bridge.pack_robot_server_message(
                    {
                        "type": "send_command",
                        "kind": robot_command_kind,
                        "actions": commands,
                        "source_hz": args.source_hz,
                    }
                )
            )
            ack = spiritai_bridge.unpack_robot_server_message(robot_ws.recv())
            if not ack.get("accepted", False):
                logging.warning("Step %d rejected by robot_server: %s", step, ack.get("error"))
                continue
            logging.info(
                "Step %d accepted: chunk_id=%s actions=%s expected_finish_at=%s",
                step,
                ack.get("chunk_id"),
                commands.shape,
                ack.get("expected_finish_at"),
            )
            previous_state = current_state
            previous_commands = commands

            if args.prefetch_next_chunk and step + 1 < args.max_steps:
                expected_finish_at = ack.get("expected_finish_at")
                if isinstance(expected_finish_at, int | float):
                    remaining_s = max(0.0, float(expected_finish_at) - time.monotonic())
                    delay_s = remaining_s * args.prefetch_delay_fraction
                    if delay_s > 0:
                        time.sleep(delay_s)
                prefetch_obs, prefetch_images = _get_robot_obs(robot_ws)
                prefetch_start_s = time.perf_counter()
                prefetched_actions = _infer_policy_actions(
                    policy,
                    prefetch_obs,
                    prefetch_images,
                    prompt=args.prompt,
                    policy_action_layout=args.policy_action_layout,
                )
                prefetch_latency_s = time.perf_counter() - prefetch_start_s
                logging.info("Step %d prefetch policy inference latency: %.3fs", step, prefetch_latency_s)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
