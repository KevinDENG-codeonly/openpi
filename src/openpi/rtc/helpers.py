"""RTC helper functions: soft mask (Eq. 5), guidance utilities.

Reference: arXiv 2506.07339, Section 3.2 - Soft-boundary Inpainting Guidance.
"""

import numpy as np


def compute_soft_mask(
    action_horizon: int,
    consumed: int,
    s_min: int,
) -> np.ndarray:
    """Compute the RTC soft inpainting mask W per Eq. 5.

    W_i = 1             for i < d           (already-committed region)
    W_i = (exp(c_i)-1)/(e-1)  for d <= i < H-f   (soft transition region)
    W_i = 0             for i >= H-f         (free/unconstrained tail)

    where:
      H = action_horizon
      d = consumed (number of already-executed steps from the previous chunk)
      s = s_min (minimum free steps at the end)
      f = max(d, s), so the mask never constrains non-overlapping zero-padded target steps
      c_i = (H - f - i) / (H - f - d + 1)  (linear schedule from ~1 to ~0)

    Args:
        action_horizon: H, total action chunk length.
        consumed: d, number of steps already consumed/executed from the previous chunk.
        s_min: s, minimum free steps at the tail (unconstrained).

    Returns:
        Mask array of shape (action_horizon,) with values in [0, 1].
    """
    horizon = action_horizon
    d = consumed
    s = s_min

    # Edge cases
    d = max(d, 0)
    s = max(s, 0)
    if d >= horizon:
        return np.zeros(horizon, dtype=np.float32)

    free_tail = max(d, s)
    free_start = horizon - free_tail

    # If consumed + the free tail covers the full horizon, there's no transition region; clamp.
    if d >= free_start:
        # All committed or overlap: mask is 1 up to d, 0 after.
        mask = np.zeros(horizon, dtype=np.float32)
        mask[: min(d, horizon)] = 1.0
        return mask

    mask = np.zeros(horizon, dtype=np.float32)

    # Region 1: fully committed (i < d)
    mask[:d] = 1.0

    # Region 2: soft transition (d <= i < H - free_tail)
    e_minus_1 = np.e - 1.0
    denom = free_start - d + 1  # denominator for c_i linear schedule
    for i in range(d, free_start):
        c_i = (free_start - i) / denom
        mask[i] = (np.exp(c_i) - 1.0) / e_minus_1

    # Region 3: free tail (i >= H - free_tail) is already 0.

    return mask


def build_rtc_target_and_mask(
    previous_actions: np.ndarray,
    consumed: int,
    s_min: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build RTC target and mask arrays for guided inference.

    Given the previous predicted action chunk and how many steps were consumed,
    constructs the target trajectory and soft mask for the next inference call.

    The target is built by shifting the previous actions: the first `consumed` actions
    have been executed and are dropped, so previous_actions[consumed:] becomes the
    target for positions [0 : H-consumed], and the tail is zero-padded. The mask
    is zero over that zero-padded non-overlap tail.

    Args:
        previous_actions: Previous action chunk, shape (H, action_dim).
        consumed: Number of steps already executed from previous_actions.
        s_min: Minimum free steps at the tail.

    Returns:
        Tuple of (target, mask):
          - target: shape (H, action_dim), the inpainting target trajectory.
          - mask: shape (H,), the soft mask weights.
    """
    horizon, action_dim = previous_actions.shape
    target = np.zeros((horizon, action_dim), dtype=np.float32)

    consumed = max(consumed, 0)

    # Shift: remaining actions from previous chunk become target
    remaining = horizon - consumed
    if remaining > 0:
        target[:remaining] = previous_actions[consumed:]

    mask = compute_soft_mask(horizon, consumed=consumed, s_min=s_min)
    return target, mask
