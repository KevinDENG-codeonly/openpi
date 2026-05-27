import dataclasses
import logging
import time

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
        raise spiritai_bridge.RobotServerProtocolError(
            f"Unexpected set_external_following response: {msg.get('type')}"
        )
    if not msg.get("accepted", False):
        raise spiritai_bridge.RobotServerProtocolError(
            f"robot_server did not enable external following: enabled={msg.get('enabled')} error={msg.get('error')}"
        )
    logging.info("Robot external following enabled: %s", msg.get("enabled"))


def _infer_policy_actions(
    policy: _websocket_client_policy.WebsocketClientPolicy,
    obs: dict,
    images: dict,
    *,
    prompt: str,
) -> object:
    policy_obs = spiritai_bridge.map_robot_server_observation(obs, images, prompt=prompt)
    return policy.infer(policy_obs)["actions"]


def main(args: Args) -> None:
    if not 0.0 <= args.prefetch_delay_fraction <= 1.0:
        raise ValueError(f"prefetch_delay_fraction must be in [0, 1], got {args.prefetch_delay_fraction}")

    policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.policy_host,
        port=args.policy_port,
    )
    logging.info("Policy server metadata: %s", policy.get_server_metadata())

    with websockets.sync.client.connect(args.robot_url, max_size=None, compression=None) as robot_ws:
        hello = spiritai_bridge.unpack_robot_server_message(robot_ws.recv())
        if hello.get("type") != "hello":
            raise spiritai_bridge.RobotServerProtocolError(f"Expected hello from robot_server, got: {hello.get('type')}")

        metadata = hello["metadata"]
        joint_command_dim = spiritai_bridge.choose_joint_command_dim(metadata)
        logging.info(
            "Robot server connected: structure=%s joint_dim=%s command_dim=%s cameras=%s",
            metadata.get("structure"),
            metadata.get("joint_dim"),
            joint_command_dim,
            metadata.get("cameras"),
        )
        if args.enable_external_following:
            _set_robot_external_following(robot_ws, enabled=True)
        if args.startup_delay_s > 0:
            logging.info("Startup delay %.1fs before first inference.", args.startup_delay_s)
            time.sleep(args.startup_delay_s)

        max_chunk = int(metadata.get("max_chunk", 60))
        motion_limits = spiritai_bridge.JointMotionLimits(
            max_arm_velocity_rad_s=args.max_arm_velocity_rad_s,
            max_torso_velocity_rad_s=args.max_torso_velocity_rad_s,
            max_gripper_velocity_s=args.max_gripper_velocity_s,
            max_base_speed=args.max_base_speed,
            max_joint_accel_rad_s2=args.max_joint_accel_rad_s2,
        )
        previous_joint_state = None
        previous_joint_commands = None
        prefetched_actions = None
        for step in range(args.max_steps):
            _wait_until_robot_idle(robot_ws, args.busy_sleep_s)
            obs, images = _get_robot_obs(robot_ws)
            current_joint_state = spiritai_bridge.robot_server_obs_to_joint_command_layout(obs, joint_command_dim)
            if previous_joint_state is not None:
                actual_delta = current_joint_state - previous_joint_state
                actual_delta_summary = spiritai_bridge.summarize_joint_delta_by_group(actual_delta, joint_command_dim)
                logging.info(
                    "Step %d actual state delta since previous chunk: %s",
                    step,
                    " ".join(
                        f"{name}=mean:{mean_abs:.4f}/max:{max_abs:.4f}"
                        for name, (mean_abs, max_abs) in actual_delta_summary.items()
                    ),
                )

            if prefetched_actions is None:
                raw_actions = _infer_policy_actions(policy, obs, images, prompt=args.prompt)
            else:
                raw_actions = prefetched_actions
                prefetched_actions = None

            joint_commands = spiritai_bridge.spiritai_actions_to_joint_commands(raw_actions, joint_command_dim)
            if joint_commands.shape[0] > max_chunk:
                joint_commands = joint_commands[:max_chunk]
            joint_commands = spiritai_bridge.suppress_chunk_start_rollback(
                joint_commands,
                previous_joint_commands,
                guard_steps=args.rollback_guard_steps,
                rollback_scale=args.rollback_scale,
            )
            joint_commands = spiritai_bridge.blend_joint_command_start(
                joint_commands,
                current_joint_state,
                args.blend_steps,
            )
            joint_commands, motion_stats = spiritai_bridge.limit_joint_command_motion(
                joint_commands,
                current_joint_state,
                source_hz=args.source_hz,
                limits=motion_limits,
                previous_commands=previous_joint_commands,
            )
            first_delta = joint_commands[0] - current_joint_state
            delta_summary = spiritai_bridge.summarize_joint_delta_by_group(first_delta, joint_command_dim)
            logging.info(
                "Step %d command stats: min=%.4f max=%.4f first_delta_mean_abs=%.4f first_delta_max_abs=%.4f",
                step,
                float(joint_commands.min()),
                float(joint_commands.max()),
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
                        "kind": "joint",
                        "actions": joint_commands,
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
                joint_commands.shape,
                ack.get("expected_finish_at"),
            )
            previous_joint_state = current_joint_state
            previous_joint_commands = joint_commands

            if args.prefetch_next_chunk and step + 1 < args.max_steps:
                expected_finish_at = ack.get("expected_finish_at")
                if isinstance(expected_finish_at, int | float):
                    remaining_s = max(0.0, float(expected_finish_at) - time.monotonic())
                    delay_s = remaining_s * args.prefetch_delay_fraction
                    if delay_s > 0:
                        time.sleep(delay_s)
                prefetch_obs, prefetch_images = _get_robot_obs(robot_ws)
                prefetched_actions = _infer_policy_actions(policy, prefetch_obs, prefetch_images, prompt=args.prompt)
                logging.info("Step %d prefetched next policy chunk", step)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
