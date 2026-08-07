from collections.abc import Mapping
from typing import Any

import numpy as np


class RTCRequestError(ValueError):
    """Raised when an RTC request is incompatible with a policy's capabilities."""


def make_capabilities(train_config: Any) -> dict[str, Any]:
    """Build the RTC capability metadata advertised by a trained policy."""
    model = train_config.model
    capabilities = {
        "algorithm": "training_time_v1" if train_config.rtc_training.enabled else "disabled",
        "model_type": model.model_type.value,
        "action_horizon": model.action_horizon,
        "action_dim": model.action_dim,
    }
    if train_config.rtc_training.enabled:
        capabilities["training_max_delay_steps"] = train_config.rtc_training.max_delay_steps
    return capabilities


def _capability_int(capability: Mapping[str, Any], name: str) -> int:
    value = capability.get(name)
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise RTCRequestError(f"RTC capability is missing a valid {name}.")
    return int(value)


def validate_training_time_request(
    request: Mapping[str, Any], capability: Mapping[str, Any] | None
) -> tuple[np.ndarray, int]:
    """Validate a training-time RTC request and return its normalized inputs."""
    if not isinstance(capability, Mapping):
        raise RTCRequestError("This policy does not advertise RTC capabilities.")
    if capability.get("algorithm") == "disabled":
        raise RTCRequestError("RTC is disabled for this policy.")
    if capability.get("algorithm") != "training_time_v1":
        raise RTCRequestError("This policy does not support the requested RTC algorithm.")

    if not isinstance(request, Mapping):
        raise RTCRequestError("RTC request must be a mapping.")
    expected_keys = {"algorithm", "action_prefix", "delay_steps"}
    if set(request) != expected_keys:
        raise RTCRequestError(f"RTC request must contain exactly these fields: {sorted(expected_keys)}.")
    if request["algorithm"] != capability["algorithm"]:
        raise RTCRequestError("RTC request algorithm does not match this policy's capability.")

    action_horizon = _capability_int(capability, "action_horizon")
    action_dim = _capability_int(capability, "action_dim")
    max_delay_steps = _capability_int(capability, "training_max_delay_steps")

    try:
        action_prefix = np.asarray(request["action_prefix"], dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise RTCRequestError("RTC action_prefix must be convertible to a float32 array.") from exc
    expected_prefix_shape = (action_horizon, action_dim)
    if action_prefix.shape != expected_prefix_shape:
        raise RTCRequestError(
            f"RTC action_prefix must have shape {expected_prefix_shape}, got {action_prefix.shape}."
        )

    delay_value = np.asarray(request["delay_steps"])
    if delay_value.ndim != 0 or not np.issubdtype(delay_value.dtype, np.integer):
        raise RTCRequestError("RTC delay_steps must be a scalar integer.")
    delay_steps = int(delay_value.item())
    if not 0 <= delay_steps <= max_delay_steps:
        raise RTCRequestError(
            f"RTC delay_steps must satisfy 0 <= delay_steps <= {max_delay_steps}, got {delay_steps}."
        )

    return action_prefix, delay_steps
