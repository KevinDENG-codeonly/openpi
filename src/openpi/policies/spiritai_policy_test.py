import numpy as np

from openpi.models import model as _model
from openpi.policies import spiritai_policy


def _base_data() -> dict:
    return {
        "cam_high": np.zeros((8, 8, 3), dtype=np.uint8),
        "cam_left_wrist": np.ones((8, 8, 3), dtype=np.uint8),
        "cam_right_wrist": np.ones((8, 8, 3), dtype=np.uint8) * 2,
        "leftarm_state_joint_pos": np.arange(0, 7, dtype=np.float32),
        "leftarm_state_cart_pos": np.arange(0, 6, dtype=np.float32),
        "leftarm_state_psi": np.array([7], dtype=np.float32),
        "leftarm_gripper_state_pos": np.array([8], dtype=np.float32),
        "rightarm_state_joint_pos": np.arange(9, 16, dtype=np.float32),
        "rightarm_state_cart_pos": np.arange(9, 15, dtype=np.float32),
        "rightarm_state_psi": np.array([16], dtype=np.float32),
        "rightarm_gripper_state_pos": np.array([17], dtype=np.float32),
        "torso_state_joint_pos": np.arange(18, 24, dtype=np.float32),
        "torso_state_cart_pos": np.arange(18, 24, dtype=np.float32),
        "base_state_speed": np.arange(24, 27, dtype=np.float32),
        "prompt": "fold the paper box",
    }


def test_spiritai_inputs_keep_joint_action_layout() -> None:
    data = _base_data()
    for i, key in enumerate(spiritai_policy.ACTION_KEYS):
        dim = 7 if "joint_pos" in key and "torso" not in key else 1
        if "torso" in key:
            dim = 6
        if "base" in key:
            dim = 3
        data[key] = np.full((2, dim), i, dtype=np.float32)

    transform = spiritai_policy.SpiritaiInputs(model_type=_model.ModelType.PI05)
    out = transform(data)

    assert out["state"].shape == (spiritai_policy.ACTION_DIM,)
    assert out["actions"].shape == (2, spiritai_policy.ACTION_DIM)
    assert out["prompt"] == "fold the paper box"


def test_spiritai_cartesian_inputs_use_cartesian_action_layout() -> None:
    data = _base_data()
    for i, key in enumerate(spiritai_policy.CARTESIAN_ACTION_KEYS):
        dim = 6 if "cart_pos" in key else 1
        if "base" in key:
            dim = 3
        data[key] = np.full((2, dim), i, dtype=np.float32)

    transform = spiritai_policy.SpiritaiCartesianInputs(model_type=_model.ModelType.PI05)
    out = transform(data)

    assert out["state"].shape == (spiritai_policy.CARTESIAN_ACTION_DIM,)
    assert out["actions"].shape == (2, spiritai_policy.CARTESIAN_ACTION_DIM)
    np.testing.assert_array_equal(out["actions"][:, :6], np.zeros((2, 6), dtype=np.float32))
    np.testing.assert_array_equal(out["actions"][:, -3:], np.full((2, 3), 7, dtype=np.float32))


def test_spiritai_cartesian_outputs_drop_model_padding() -> None:
    actions = np.ones((50, 32), dtype=np.float32)

    out = spiritai_policy.SpiritaiCartesianOutputs()({"actions": actions})

    assert out["actions"].shape == (50, spiritai_policy.CARTESIAN_ACTION_DIM)
