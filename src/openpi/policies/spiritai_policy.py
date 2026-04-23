"""Transforms for Spirit AI humanoid robot datasets.

The Spirit AI moz1 robot has:
- Left arm: 7 joint pos + 1 psi + 1 gripper = 9 dims
- Right arm: 7 joint pos + 1 psi + 1 gripper = 9 dims
- Torso: 6 joint pos = 6 dims
- Base: 3 speed = 3 dims
Total state/action dimension: 27

Cameras: cam_high (overhead), cam_left_wrist, cam_right_wrist

State uses *_state_* keys, actions use *_cmd_* keys.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# Ordered list of dataset columns that compose the state vector.
STATE_KEYS = [
    "leftarm_state_joint_pos",   # (7,)
    "leftarm_state_psi",         # (1,)
    "leftarm_gripper_state_pos", # (1,)
    "rightarm_state_joint_pos",  # (7,)
    "rightarm_state_psi",        # (1,)
    "rightarm_gripper_state_pos",# (1,)
    "torso_state_joint_pos",     # (6,)
    "base_state_speed",          # (3,)
]

# Ordered list of dataset columns that compose the action vector (must match state order).
ACTION_KEYS = [
    "leftarm_cmd_joint_pos",   # (7,)
    "leftarm_cmd_psi",         # (1,)
    "leftarm_gripper_cmd_pos", # (1,)
    "rightarm_cmd_joint_pos",  # (7,)
    "rightarm_cmd_psi",        # (1,)
    "rightarm_gripper_cmd_pos",# (1,)
    "torso_cmd_joint_pos",     # (6,)
    "base_cmd_speed",          # (3,)
]

# Total action dimension after concatenation.
ACTION_DIM = 27


def make_spiritai_example() -> dict:
    """Creates a random input example for the Spirit AI policy."""
    return {
        "cam_high": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "cam_left_wrist": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "cam_right_wrist": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "leftarm_state_joint_pos": np.random.rand(7).astype(np.float32),
        "leftarm_state_psi": np.random.rand(1).astype(np.float32),
        "leftarm_gripper_state_pos": np.random.rand(1).astype(np.float32),
        "rightarm_state_joint_pos": np.random.rand(7).astype(np.float32),
        "rightarm_state_psi": np.random.rand(1).astype(np.float32),
        "rightarm_gripper_state_pos": np.random.rand(1).astype(np.float32),
        "torso_state_joint_pos": np.random.rand(6).astype(np.float32),
        "base_state_speed": np.random.rand(3).astype(np.float32),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class SpiritaiInputs(transforms.DataTransformFn):
    """Converts Spirit AI dataset fields into the model input format.

    Concatenates individual joint/gripper/torso/base columns into a single state vector,
    handles image parsing, and passes through prompt and actions.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Concatenate individual state columns into one state vector (27-dim).
        state_parts = [np.asarray(data[k]).flatten() for k in STATE_KEYS]
        state = np.concatenate(state_parts)

        # Parse images.
        base_image = _parse_image(data["cam_high"])
        left_wrist = _parse_image(data["cam_left_wrist"])
        right_wrist = _parse_image(data["cam_right_wrist"])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist,
                "right_wrist_0_rgb": right_wrist,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # All three cameras are real for Spirit AI robot.
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Concatenate action columns into combined action array.
        # During training, each action key is a (action_horizon, dim) array from LeRobot delta_timestamps.
        # Concatenate along the last axis to get (action_horizon, 27).
        if ACTION_KEYS[0] in data:
            action_parts = [np.asarray(data[k]) for k in ACTION_KEYS]
            actions = np.concatenate(action_parts, axis=-1)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class SpiritaiOutputs(transforms.DataTransformFn):
    """Extracts Spirit AI actions from padded model output.

    The model output is padded to 32 dims; we return only the first 27.
    """

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :ACTION_DIM])}
