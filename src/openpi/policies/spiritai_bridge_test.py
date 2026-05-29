import numpy as np
import pytest

from openpi.policies import spiritai_bridge


def test_spiritai_actions_to_25d_joint_commands_drops_psi_only() -> None:
    actions = np.arange(2 * spiritai_bridge.ACTION_DIM, dtype=np.float32).reshape(2, spiritai_bridge.ACTION_DIM)

    commands = spiritai_bridge.spiritai_actions_to_joint_commands(actions, 25)

    expected = np.concatenate(
        [
            actions[:, 0:7],
            actions[:, 8:9],
            actions[:, 9:16],
            actions[:, 17:18],
            actions[:, 18:24],
            actions[:, 24:27],
        ],
        axis=-1,
    )
    np.testing.assert_array_equal(commands, expected)
    assert commands.shape == (2, 25)


def test_spiritai_actions_to_22d_joint_commands_drops_psi_and_base() -> None:
    actions = np.arange(spiritai_bridge.ACTION_DIM, dtype=np.float32).reshape(1, spiritai_bridge.ACTION_DIM)

    commands = spiritai_bridge.spiritai_actions_to_joint_commands(actions, 22)

    expected = np.concatenate(
        [
            actions[:, 0:7],
            actions[:, 8:9],
            actions[:, 9:16],
            actions[:, 17:18],
            actions[:, 18:24],
        ],
        axis=-1,
    )
    np.testing.assert_array_equal(commands, expected)
    assert commands.shape == (1, 22)


def test_spiritai_cartesian_actions_to_23d_cartesian_commands_drops_psi_only() -> None:
    actions = np.arange(2 * spiritai_bridge.CARTESIAN_ACTION_DIM, dtype=np.float32).reshape(
        2, spiritai_bridge.CARTESIAN_ACTION_DIM
    )

    commands = spiritai_bridge.spiritai_cartesian_actions_to_cartesian_commands(actions, 23)

    expected = np.concatenate(
        [
            actions[:, 0:6],
            actions[:, 7:8],
            actions[:, 8:14],
            actions[:, 15:16],
            actions[:, 16:22],
            actions[:, 22:25],
        ],
        axis=-1,
    )
    np.testing.assert_array_equal(commands, expected)
    assert commands.shape == (2, 23)


def test_spiritai_cartesian_actions_to_20d_cartesian_commands_drops_psi_and_base() -> None:
    actions = np.arange(spiritai_bridge.CARTESIAN_ACTION_DIM, dtype=np.float32).reshape(
        1, spiritai_bridge.CARTESIAN_ACTION_DIM
    )

    commands = spiritai_bridge.spiritai_cartesian_actions_to_cartesian_commands(actions, 20)

    expected = np.concatenate(
        [
            actions[:, 0:6],
            actions[:, 7:8],
            actions[:, 8:14],
            actions[:, 15:16],
            actions[:, 16:22],
        ],
        axis=-1,
    )
    np.testing.assert_array_equal(commands, expected)
    assert commands.shape == (1, 20)


def test_map_robot_server_observation_uses_required_cameras_and_state_keys() -> None:
    obs = {
        "leftarm_state_joint_pos": np.ones(7),
        "leftarm_state_psi": np.ones(1) * 2,
        "leftarm_gripper_state_pos": np.ones(1) * 3,
        "rightarm_state_joint_pos": np.ones(7) * 4,
        "rightarm_state_psi": np.ones(1) * 5,
        "rightarm_gripper_state_pos": np.ones(1) * 6,
        "torso_state_joint_pos": np.ones(6) * 7,
        "base_state_speed": np.ones(3) * 8,
        "ignored": np.ones(1),
    }
    images = {
        "cam_high": np.zeros((240, 320, 3), dtype=np.uint8),
        "cam_high_extra": np.ones((240, 320, 3), dtype=np.uint8),
        "cam_left_wrist": np.ones((240, 320, 3), dtype=np.uint8) * 2,
        "cam_left_wrist_extra": np.ones((240, 320, 3), dtype=np.uint8) * 3,
        "cam_right_wrist": np.ones((240, 320, 3), dtype=np.uint8) * 4,
        "cam_right_wrist_extra": np.ones((240, 320, 3), dtype=np.uint8) * 5,
    }

    policy_obs = spiritai_bridge.map_robot_server_observation(obs, images, prompt="fold the paper box")

    assert set(spiritai_bridge.CAMERA_KEYS).issubset(policy_obs)
    assert policy_obs["prompt"] == "fold the paper box"
    assert policy_obs["cam_high"].shape == (240, 320, 3)
    assert policy_obs["leftarm_state_joint_pos"].dtype == np.float32
    assert policy_obs["base_state_speed"].shape == (3,)


def test_map_robot_server_cartesian_observation_uses_cartesian_state_keys() -> None:
    obs = {
        "leftarm_state_cart_pos": np.ones(6),
        "leftarm_state_joint_pos": np.ones(7) * 100,
        "leftarm_state_psi": np.ones(1) * 2,
        "leftarm_gripper_state_pos": np.ones(1) * 3,
        "rightarm_state_cart_pos": np.ones(6) * 4,
        "rightarm_state_joint_pos": np.ones(7) * 200,
        "rightarm_state_psi": np.ones(1) * 5,
        "rightarm_gripper_state_pos": np.ones(1) * 6,
        "torso_state_cart_pos": np.ones(6) * 7,
        "torso_state_joint_pos": np.ones(6) * 300,
        "base_state_speed": np.ones(3) * 8,
    }
    images = {
        "cam_high": np.zeros((240, 320, 3), dtype=np.uint8),
        "cam_left_wrist": np.ones((240, 320, 3), dtype=np.uint8) * 2,
        "cam_right_wrist": np.ones((240, 320, 3), dtype=np.uint8) * 4,
    }

    policy_obs = spiritai_bridge.map_robot_server_cartesian_observation(obs, images, prompt="fold the paper box")

    assert set(spiritai_bridge.CAMERA_KEYS).issubset(policy_obs)
    assert policy_obs["prompt"] == "fold the paper box"
    assert policy_obs["leftarm_state_cart_pos"].dtype == np.float32
    assert policy_obs["leftarm_state_cart_pos"].shape == (6,)
    assert "leftarm_state_joint_pos" not in policy_obs


def test_choose_joint_command_dim_prefers_widest_supported_layout() -> None:
    assert spiritai_bridge.choose_joint_command_dim({"joint_dim": 25, "accepted_joint_dims": [16, 22, 25]}) == 25
    assert spiritai_bridge.choose_joint_command_dim({"joint_dim": 25, "accepted_joint_dims": [16, 22]}) == 22
    assert spiritai_bridge.choose_joint_command_dim({"joint_dim": 22, "accepted_joint_dims": [16, 22, 25]}) == 22


def test_choose_joint_command_dim_rejects_unsupported_metadata() -> None:
    with pytest.raises(spiritai_bridge.RobotServerProtocolError):
        spiritai_bridge.choose_joint_command_dim({"joint_dim": 14, "accepted_joint_dims": [14]})


def test_choose_cartesian_command_dim_prefers_widest_supported_layout() -> None:
    assert spiritai_bridge.choose_cartesian_command_dim({"cart_dim": 23, "accepted_cart_dims": [14, 20, 23]}) == 23
    assert spiritai_bridge.choose_cartesian_command_dim({"cart_dim": 23, "accepted_cart_dims": [14, 20]}) == 20
    assert spiritai_bridge.choose_cartesian_command_dim({"cart_dim": 20, "accepted_cart_dims": [14, 20, 23]}) == 20


def test_choose_cartesian_command_dim_rejects_unsupported_metadata() -> None:
    with pytest.raises(spiritai_bridge.RobotServerProtocolError):
        spiritai_bridge.choose_cartesian_command_dim({"cart_dim": 13, "accepted_cart_dims": [13]})


def test_robot_server_message_codec_round_trips_ndarray() -> None:
    actions = np.arange(50, dtype=np.float32).reshape(2, 25)

    packed = spiritai_bridge.pack_robot_server_message({"type": "send_command", "actions": actions})
    unpacked = spiritai_bridge.unpack_robot_server_message(packed)

    assert unpacked["type"] == "send_command"
    assert isinstance(unpacked["actions"], np.ndarray)
    np.testing.assert_array_equal(unpacked["actions"], actions)


def test_robot_server_obs_to_25d_joint_command_layout() -> None:
    obs = {
        "leftarm_state_joint_pos": np.arange(0, 7, dtype=np.float32),
        "leftarm_gripper_state_pos": np.array([7], dtype=np.float32),
        "rightarm_state_joint_pos": np.arange(8, 15, dtype=np.float32),
        "rightarm_gripper_state_pos": np.array([15], dtype=np.float32),
        "torso_state_joint_pos": np.arange(16, 22, dtype=np.float32),
        "base_state_speed": np.arange(22, 25, dtype=np.float32),
    }

    state = spiritai_bridge.robot_server_obs_to_joint_command_layout(obs, 25)

    np.testing.assert_array_equal(state, np.arange(25, dtype=np.float32))


def test_robot_server_obs_to_23d_cartesian_command_layout() -> None:
    obs = {
        "leftarm_state_cart_pos": np.arange(0, 6, dtype=np.float32),
        "leftarm_gripper_state_pos": np.array([6], dtype=np.float32),
        "rightarm_state_cart_pos": np.arange(7, 13, dtype=np.float32),
        "rightarm_gripper_state_pos": np.array([13], dtype=np.float32),
        "torso_state_cart_pos": np.arange(14, 20, dtype=np.float32),
        "base_state_speed": np.arange(20, 23, dtype=np.float32),
    }

    state = spiritai_bridge.robot_server_obs_to_cartesian_command_layout(obs, 23)

    np.testing.assert_array_equal(state, np.arange(23, dtype=np.float32))


def test_summarize_joint_delta_by_group() -> None:
    delta = np.arange(25, dtype=np.float32)

    summary = spiritai_bridge.summarize_joint_delta_by_group(delta, 25)

    assert summary["left_arm"] == (3.0, 6.0)
    assert summary["left_gripper"] == (7.0, 7.0)
    assert summary["right_arm"] == (11.0, 14.0)
    assert summary["right_gripper"] == (15.0, 15.0)
    assert summary["torso"] == (18.5, 21.0)
    assert summary["base"] == (23.0, 24.0)


def test_summarize_cartesian_delta_by_group() -> None:
    delta = np.arange(23, dtype=np.float32)

    summary = spiritai_bridge.summarize_cartesian_delta_by_group(delta, 23)

    assert summary["left_translation"] == (1.0, 2.0)
    assert summary["left_rotation"] == (4.0, 5.0)
    assert summary["left_gripper"] == (6.0, 6.0)
    assert summary["right_translation"] == (8.0, 9.0)
    assert summary["right_rotation"] == (11.0, 12.0)
    assert summary["right_gripper"] == (13.0, 13.0)
    assert summary["torso_translation"] == (15.0, 16.0)
    assert summary["torso_rotation"] == (18.0, 19.0)
    assert summary["base"] == (21.0, 22.0)


def test_blend_joint_command_start_aligns_first_frame_and_blends_prefix() -> None:
    commands = np.ones((5, 3), dtype=np.float32)
    current_state = np.array([10.0, 20.0, 30.0], dtype=np.float32)

    blended = spiritai_bridge.blend_joint_command_start(commands, current_state, blend_steps=4)

    np.testing.assert_array_equal(blended[0], current_state)
    np.testing.assert_allclose(blended[1], current_state * (2.0 / 3.0) + commands[1] * (1.0 / 3.0), atol=1e-6)
    np.testing.assert_allclose(blended[2], current_state * (1.0 / 3.0) + commands[2] * (2.0 / 3.0), atol=1e-6)
    np.testing.assert_array_equal(blended[3], commands[3])
    np.testing.assert_array_equal(blended[4], commands[4])


def test_suppress_chunk_start_rollback_scales_opposite_arm_and_torso_motion() -> None:
    previous = np.zeros((2, 25), dtype=np.float32)
    previous[-2, 0] = 0.0
    previous[-1, 0] = 1.0
    previous[-2, 16] = 0.0
    previous[-1, 16] = -1.0

    commands = previous[-1][None, :].repeat(5, axis=0)
    commands[:4, 0] = 0.5
    commands[:4, 7] = 0.5
    commands[:4, 16] = -0.5
    commands[4, 0] = 0.25

    guarded = spiritai_bridge.suppress_chunk_start_rollback(
        commands,
        previous,
        guard_steps=4,
        rollback_scale=0.2,
    )

    np.testing.assert_allclose(guarded[:4, 0], 0.9)
    np.testing.assert_allclose(guarded[:4, 16], -0.9)
    np.testing.assert_allclose(guarded[:4, 7], 0.5)
    assert guarded[4, 0] == commands[4, 0]


def test_suppress_chunk_start_rollback_keeps_same_direction_motion() -> None:
    previous = np.zeros((2, 25), dtype=np.float32)
    previous[-1, 0] = 1.0
    commands = previous[-1][None, :].repeat(2, axis=0)
    commands[:, 0] = 1.5

    guarded = spiritai_bridge.suppress_chunk_start_rollback(
        commands,
        previous,
        guard_steps=2,
        rollback_scale=0.2,
    )

    np.testing.assert_array_equal(guarded, commands)


def test_suppress_cartesian_chunk_start_rollback_scales_opposite_pose_motion() -> None:
    previous = np.zeros((2, 23), dtype=np.float32)
    previous[-1, 0] = 1.0
    previous[-2, 14] = 0.0
    previous[-1, 14] = -1.0

    commands = previous[-1][None, :].repeat(5, axis=0)
    commands[:4, 0] = 0.5
    commands[:4, 6] = 0.5
    commands[:4, 14] = -0.5

    guarded = spiritai_bridge.suppress_cartesian_chunk_start_rollback(
        commands,
        previous,
        guard_steps=4,
        rollback_scale=0.2,
    )

    np.testing.assert_allclose(guarded[:4, 0], 0.9)
    np.testing.assert_allclose(guarded[:4, 14], -0.9)
    np.testing.assert_allclose(guarded[:4, 6], 0.5)


def test_limit_joint_command_motion_limits_arm_velocity_from_current_state() -> None:
    commands = np.zeros((3, 25), dtype=np.float32)
    commands[:, 0] = 1.0
    current_state = np.zeros(25, dtype=np.float32)

    limited, stats = spiritai_bridge.limit_joint_command_motion(
        commands,
        current_state,
        source_hz=10.0,
        limits=spiritai_bridge.JointMotionLimits(
            max_arm_velocity_rad_s=0.2,
            max_torso_velocity_rad_s=1.0,
            max_gripper_velocity_s=1.0,
            max_base_speed=1.0,
        ),
    )

    np.testing.assert_allclose(limited[:, 0], np.array([0.02, 0.04, 0.06], dtype=np.float32), atol=1e-6)
    assert stats["max_raw_velocity"] == pytest.approx(10.0)
    assert stats["max_limited_velocity"] == pytest.approx(0.2)
    assert stats["limited_fraction"] > 0.0


def test_limit_joint_command_motion_uses_group_specific_limits_and_clamps_base_speed() -> None:
    commands = np.zeros((2, 25), dtype=np.float32)
    commands[:, 0] = 1.0
    commands[:, 7] = 1.0
    commands[:, 16] = 1.0
    commands[:, 22:25] = 10.0
    current_state = np.zeros(25, dtype=np.float32)

    limited, _ = spiritai_bridge.limit_joint_command_motion(
        commands,
        current_state,
        source_hz=10.0,
        limits=spiritai_bridge.JointMotionLimits(
            max_arm_velocity_rad_s=0.3,
            max_torso_velocity_rad_s=0.1,
            max_gripper_velocity_s=0.5,
            max_base_speed=0.05,
        ),
    )

    assert limited[0, 0] == pytest.approx(0.03)
    assert limited[0, 7] == pytest.approx(0.05)
    assert limited[0, 16] == pytest.approx(0.01)
    np.testing.assert_allclose(limited[:, 22:25], 0.05)


def test_limit_joint_command_motion_can_limit_joint_acceleration() -> None:
    commands = np.ones((3, 25), dtype=np.float32)
    current_state = np.zeros(25, dtype=np.float32)

    limited, stats = spiritai_bridge.limit_joint_command_motion(
        commands,
        current_state,
        source_hz=10.0,
        limits=spiritai_bridge.JointMotionLimits(
            max_arm_velocity_rad_s=1.0,
            max_torso_velocity_rad_s=1.0,
            max_gripper_velocity_s=1.0,
            max_base_speed=1.0,
            max_joint_accel_rad_s2=0.5,
        ),
    )

    np.testing.assert_allclose(limited[:, 0], np.array([0.005, 0.015, 0.03], dtype=np.float32), atol=1e-6)
    assert stats["max_limited_velocity"] == pytest.approx(0.15)


def test_limit_cartesian_command_motion_uses_group_specific_limits_and_clamps_base_speed() -> None:
    commands = np.zeros((2, 23), dtype=np.float32)
    commands[:, 0] = 1.0
    commands[:, 3] = 1.0
    commands[:, 6] = 1.0
    commands[:, 14] = 1.0
    commands[:, 17] = 1.0
    commands[:, 20:23] = 10.0
    current_state = np.zeros(23, dtype=np.float32)

    limited, _ = spiritai_bridge.limit_cartesian_command_motion(
        commands,
        current_state,
        source_hz=10.0,
        limits=spiritai_bridge.CartesianMotionLimits(
            max_arm_translation_m_s=0.1,
            max_arm_rotation_rad_s=0.3,
            max_torso_translation_m_s=0.05,
            max_torso_rotation_rad_s=0.2,
            max_gripper_velocity_s=0.5,
            max_base_speed=0.05,
        ),
    )

    assert limited[0, 0] == pytest.approx(0.01)
    assert limited[0, 3] == pytest.approx(0.03)
    assert limited[0, 6] == pytest.approx(0.05)
    assert limited[0, 14] == pytest.approx(0.005)
    assert limited[0, 17] == pytest.approx(0.02)
    np.testing.assert_allclose(limited[:, 20:23], 0.05)
