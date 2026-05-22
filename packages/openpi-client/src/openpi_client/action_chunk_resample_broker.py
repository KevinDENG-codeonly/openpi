from typing import Dict

import numpy as np
from typing_extensions import override
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation, Slerp

from openpi_client import base_policy as _base_policy
import logging

ACTION_SPLITS = [
    ("leftarm_cmd_joint_pos", 0, 7),
    ("leftarm_cmd_psi", 7, 8),
    ("leftarm_gripper_cmd_pos", 8, 9),
    ("rightarm_cmd_joint_pos", 9, 16),
    ("rightarm_cmd_psi", 16, 17),
    ("rightarm_gripper_cmd_pos", 17, 18),
    ("torso_cmd_joint_pos", 18, 24),
    ("base_cmd_speed", 24, 27),
]


def split_actions(actions: np.ndarray) -> Dict[str, np.ndarray]:
    """Split a flat action array of shape (N, 27) into a key-value dict.

    Each value has shape (N, dim) where dim matches the corresponding ACTION_KEY.
    """
    result = {}
    for key, start, end in ACTION_SPLITS:
        result[key] = actions[:, start:end]
    return result


class ActionChunkResampleBroker(_base_policy.BasePolicy):
    """Wraps a policy to resample action chunks.

    Handles the SpiritAI case where the policy returns {"actions": (N, 27)}.
    The flat 27-dim action vector is first split into named key-value pairs
    (matching ACTION_KEYS), then resampled from the model's output frequency
    to the robot's control frequency using PchipInterpolator (and SLERP for
    Cartesian rotation vectors).
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        *,
        action_horizon: int,
        resample_ratio: float = 1.0,
    ):
        self._policy = policy
        self._action_horizon = action_horizon
        self._resample_ratio = resample_ratio

        self._cur_step: int = 0
        self._last_results: Dict[str, np.ndarray] | None = None

    def _interp_impl(self, data, time_src, time_tgt, rotary=False):
        if rotary:
            data = Rotation.from_rotvec(data)
            slerp = Slerp(time_src, data)
            data = slerp(time_tgt)
            data = data.as_rotvec()
        else:
            data_interp = []
            for d in data.T:
                interp = PchipInterpolator(time_src, d)
                data_interp.append(interp(time_tgt))
            data = np.stack(data_interp, axis=-1)
        return data.astype(np.float32)

    def _resample_action(self, actions, init_action, resample_ratio):
        for key, value in actions.items():
            n_src = len(value) + 1
            break

        time_src = np.linspace(0, n_src - 1, n_src)
        time_tgt = np.arange(0, time_src[-1] + 1e-8, 1.0 / resample_ratio)

        result = [{} for _ in range(len(time_tgt))]
        for key in actions.keys():
            state_key = key.replace("cmd_", "state_")
            if state_key in init_action:
                init_val = np.atleast_2d(np.asarray(init_action[state_key], dtype=np.float32).flatten())
                data = np.vstack([init_val, actions[key]])
            else:
                data = actions[key]

            if key.endswith("cart_pos"):
                cart_pos = data[:, :3]
                cart_rot = data[:, 3:]
                cart_pos = self._interp_impl(cart_pos, time_src, time_tgt, rotary=False)
                cart_rot = self._interp_impl(cart_rot, time_src, time_tgt, rotary=True)
                data = np.concatenate([cart_pos, cart_rot], axis=1)
            else:
                data = self._interp_impl(data, time_src, time_tgt, rotary=False)

            for res, x in zip(result, data, strict=False):
                res[key] = x

        return result

    @override
    def infer(self, obs: Dict) -> Dict:
        if self._last_results is None:
            raw_results = self._policy.infer(obs)

            actions_array = raw_results["actions"]
            if isinstance(actions_array, np.ndarray) and actions_array.ndim == 2:
                split = split_actions(actions_array)
            elif isinstance(actions_array, dict):
                split = actions_array
            else:
                raise TypeError(
                    f"Expected actions to be ndarray(2D) or dict, got {type(actions_array)}"
                )

            self._last_results = {"actions": self._resample_action(split, obs, self._resample_ratio)[1:]}
            self._cur_step = 0

            if "policy_timing" in raw_results:
                logging.debug(f"policy inference time: {raw_results['policy_timing']}")

        results = self._last_results["actions"][self._cur_step]
        self._cur_step += 1

        if self._cur_step >= self._action_horizon:
            self._last_results = None

        return results

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._last_results = None
        self._cur_step = 0
