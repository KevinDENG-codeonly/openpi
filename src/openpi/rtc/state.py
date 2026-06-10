"""RTC (Real-Time Chunking) state controller for runtime execution.

Maintains the rolling action buffer, tracks consumed steps, and builds
the target/mask for each inference call.

Reference: arXiv 2506.07339, Section 3.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np

from openpi.rtc.helpers import build_rtc_target_and_mask

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class RTCState:
    """Mutable RTC execution state.

    Attributes:
        action_horizon: H, the model's action chunk length (in model action space).
        action_dim: D, model action dimension.
        s_min: Minimum free steps at the tail of the mask.
        beta: Guidance strength multiplier.
        initial_delay_steps: Number of steps to run without RTC at the start.
        previous_actions: Last predicted action chunk in model space, shape (H, D).
        consumed: Steps consumed from previous_actions since it was predicted.
        total_inferences: Counter of total inferences performed.
    """

    action_horizon: int
    action_dim: int
    s_min: int = 5
    beta: float = 0.8
    initial_delay_steps: int = 1
    previous_actions: np.ndarray | None = None
    consumed: int = 0
    total_inferences: int = 0

    def get_rtc_kwargs(self) -> dict | None:
        """Build RTC kwargs dict for Policy.infer, or None if RTC should not be applied yet."""
        if self.previous_actions is None:
            return None
        if self.total_inferences < self.initial_delay_steps:
            return None
        target, mask = build_rtc_target_and_mask(
            self.previous_actions, consumed=self.consumed, s_min=self.s_min
        )
        return {
            "target": target,
            "mask": mask,
            "beta": self.beta,
        }

    def update_after_inference(self, new_actions: np.ndarray) -> None:
        """Record the new action chunk after inference.

        Args:
            new_actions: Predicted actions in model action space, shape (H, D).
        """
        assert new_actions.shape == (self.action_horizon, self.action_dim), (
            f"Expected ({self.action_horizon}, {self.action_dim}), got {new_actions.shape}"
        )
        self.previous_actions = new_actions.copy()
        self.consumed = 0
        self.total_inferences += 1

    def mark_consumed(self, steps: int) -> None:
        """Mark steps as consumed/executed from the current chunk."""
        self.consumed += steps
