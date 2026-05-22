import time

import numpy as np
from typing_extensions import override

from openpi_client.runtime import environment as _environment
from mozrobot import MOZ1Robot, MOZ1RobotConfig


class SpiritaiMoz1Environment(_environment.Environment):
    """An environment for a MOZ1 robot on real hardware, adapted for SpiritAI policy."""

    def __init__(
        self,
        realsense_serials: str = "230322270398,313522302626,230422271253",
        camera_resolutions: str = "320*240,320*240,320*240",
        structure: str = "wholebody",
        prompt: str = "fold the paper box",
    ) -> None:
        config = MOZ1RobotConfig(
            realsense_serials=realsense_serials,
            camera_resolutions=camera_resolutions,
            structure=structure,
            robot_control_hz=120,
        )
        self.robot = MOZ1Robot(config)
        self._prompt = prompt

        self.robot.connect()

        if self.robot.is_robot_connected:
            print("Connected to MOZ1 robot successfully")
        else:
            raise RuntimeError("Failed to connect to MOZ1 robot")

    @override
    def get_observation(self) -> dict:
        obs = self.robot.capture_observation()

        obs["leftarm_gripper_state_pos"] = obs["leftarm_gripper_state_pos"].item()
        obs["rightarm_gripper_state_pos"] = obs["rightarm_gripper_state_pos"].item()

        obs["leftarm_state_psi"] = np.zeros(1, dtype=np.float32)
        obs["rightarm_state_psi"] = np.zeros(1, dtype=np.float32)

        obs["prompt"] = self._prompt

        return obs

    @override
    def apply_action(self, action: dict) -> None:
        self.robot.send_action(action)

    @override
    def is_episode_complete(self) -> bool:
        return False

    @override
    def reset(self) -> None:
        self.robot.reset()
        time.sleep(5)
        self.robot.enable_external_following_mode()
