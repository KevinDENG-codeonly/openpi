import numpy as np
import pytest

from openpi.models import model as model_module
from openpi.models import pi0_config
from openpi.policies import policy as policy_module
from openpi.policies import policy_config
from openpi.rtc.capabilities import RTCRequestError
from openpi.rtc.capabilities import make_capabilities
from openpi.rtc.capabilities import validate_training_time_request
from openpi.training import config as training_config


def _train_config(*, enabled: bool, pi05: bool = True) -> training_config.TrainConfig:
    return training_config.TrainConfig(
        name="test",
        exp_name="test",
        model=pi0_config.Pi0Config(pi05=pi05, action_horizon=4, action_dim=3),
        rtc_training=training_config.RTCTrainingConfig(enabled=enabled, max_delay_steps=2 if enabled else 0),
    )


def _training_time_request(*, delay_steps: int = 2) -> dict:
    return {
        "algorithm": "training_time_v1",
        "action_prefix": np.ones((4, 3), dtype=np.float64),
        "delay_steps": delay_steps,
    }


def test_make_capabilities_for_enabled_pi05_training() -> None:
    capabilities = make_capabilities(_train_config(enabled=True))

    assert capabilities == {
        "algorithm": "training_time_v1",
        "model_type": "pi05",
        "action_horizon": 4,
        "action_dim": 3,
        "training_max_delay_steps": 2,
    }


def test_make_capabilities_for_disabled_training_uses_model_type() -> None:
    capabilities = make_capabilities(_train_config(enabled=False, pi05=False))

    assert capabilities == {
        "algorithm": "disabled",
        "model_type": "pi0",
        "action_horizon": 4,
        "action_dim": 3,
    }


def test_validate_training_time_request_converts_accepted_prefix_to_float32() -> None:
    prefix, delay_steps = validate_training_time_request(
        _training_time_request(), make_capabilities(_train_config(enabled=True))
    )

    assert prefix.shape == (4, 3)
    assert prefix.dtype == np.float32
    assert delay_steps == 2


@pytest.mark.parametrize("non_finite_value", [np.nan, np.inf, -np.inf])
def test_validate_training_time_request_rejects_non_finite_prefix(non_finite_value: float) -> None:
    request = _training_time_request()
    request["action_prefix"][0, 0] = non_finite_value

    with pytest.raises(RTCRequestError, match="finite"):
        validate_training_time_request(request, make_capabilities(_train_config(enabled=True)))


def test_validate_training_time_request_rejects_delay_over_capability() -> None:
    with pytest.raises(RTCRequestError, match="delay_steps"):
        validate_training_time_request(
            _training_time_request(delay_steps=3), make_capabilities(_train_config(enabled=True))
        )


@pytest.mark.parametrize(
    "capability",
    [
        None,
        make_capabilities(_train_config(enabled=False)),
    ],
)
def test_validate_training_time_request_rejects_missing_or_disabled_capability(capability: dict | None) -> None:
    with pytest.raises(RTCRequestError):
        validate_training_time_request(_training_time_request(), capability)


@pytest.mark.parametrize(
    "rtc_request",
    [
        {**_training_time_request(), "algorithm": "unknown"},
        {**_training_time_request(), "unexpected": "field"},
        {"algorithm": "training_time_v1", "action_prefix": np.ones((4, 3), dtype=np.float32)},
    ],
)
def test_validate_training_time_request_rejects_unknown_or_non_exact_envelopes(rtc_request: dict) -> None:
    with pytest.raises(RTCRequestError):
        validate_training_time_request(rtc_request, make_capabilities(_train_config(enabled=True)))


@pytest.mark.parametrize("algorithm", [np.array(["training_time_v1", "training_time_v1"]), 1])
def test_validate_training_time_request_rejects_non_string_request_algorithm(algorithm) -> None:
    request = _training_time_request()
    request["algorithm"] = algorithm

    with pytest.raises(RTCRequestError, match="algorithm must be a string"):
        validate_training_time_request(request, make_capabilities(_train_config(enabled=True)))


@pytest.mark.parametrize("algorithm", [np.array(["training_time_v1", "training_time_v1"]), 1])
def test_validate_training_time_request_rejects_non_string_capability_algorithm(algorithm) -> None:
    capability = make_capabilities(_train_config(enabled=True))
    capability["algorithm"] = algorithm

    with pytest.raises(RTCRequestError, match="algorithm must be a string"):
        validate_training_time_request(_training_time_request(), capability)


class _NonConvertibleDelay:
    def __array__(self, dtype=None):
        raise TypeError("cannot convert delay")


@pytest.mark.parametrize("delay_steps", [[[0], [1, 2]], _NonConvertibleDelay()])
def test_validate_training_time_request_converts_delay_errors_to_rtc_request_errors(delay_steps) -> None:
    with pytest.raises(RTCRequestError, match="delay_steps"):
        validate_training_time_request(
            _training_time_request(delay_steps=delay_steps), make_capabilities(_train_config(enabled=True))
        )


def test_validate_training_time_request_rejects_wrong_prefix_shape() -> None:
    request = _training_time_request()
    request["action_prefix"] = np.ones((3, 3), dtype=np.float32)

    with pytest.raises(RTCRequestError, match="action_prefix"):
        validate_training_time_request(request, make_capabilities(_train_config(enabled=True)))


def test_create_trained_policy_preserves_metadata_and_adds_capabilities(monkeypatch, tmp_path) -> None:
    train_config = _train_config(enabled=True)
    original_metadata = {"existing": "value"}
    train_config = training_config.dataclasses.replace(train_config, policy_metadata=original_metadata)
    captured: dict = {}

    class CapturingPolicy:
        def __init__(self, *args, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(model_module, "restore_params", lambda *args, **kwargs: object())
    monkeypatch.setattr(pi0_config.Pi0Config, "load", lambda *args, **kwargs: object())
    monkeypatch.setattr(policy_module, "Policy", CapturingPolicy)

    policy_config.create_trained_policy(train_config, tmp_path, norm_stats={})

    assert captured["metadata"] == {
        "existing": "value",
        "rtc_capabilities": make_capabilities(train_config),
    }
    assert train_config.policy_metadata == {"existing": "value"}


def test_create_trained_jax_policy_preserves_explicit_rtc_capabilities(monkeypatch, tmp_path) -> None:
    supplied_capability = {"algorithm": "caller_supplied"}
    original_metadata = {"existing": "value", "rtc_capabilities": supplied_capability}
    train_config = training_config.dataclasses.replace(
        _train_config(enabled=True), policy_metadata=original_metadata
    )
    captured: dict = {}

    class CapturingPolicy:
        def __init__(self, *args, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(model_module, "restore_params", lambda *args, **kwargs: object())
    monkeypatch.setattr(pi0_config.Pi0Config, "load", lambda *args, **kwargs: object())
    monkeypatch.setattr(policy_module, "Policy", CapturingPolicy)

    policy_config.create_trained_policy(train_config, tmp_path, norm_stats={})

    assert captured["metadata"] == original_metadata
    assert train_config.policy_metadata == original_metadata


def test_create_trained_pytorch_policy_forces_disabled_rtc_capabilities(monkeypatch, tmp_path) -> None:
    supplied_capability = {"algorithm": "training_time_v1", "untrusted": True}
    original_metadata = {"existing": "value", "rtc_capabilities": supplied_capability}
    train_config = training_config.dataclasses.replace(
        _train_config(enabled=True), policy_metadata=original_metadata
    )
    captured: dict = {}

    class CapturingPolicy:
        def __init__(self, *args, **kwargs) -> None:
            captured.update(kwargs)

    class FakePaliGemma:
        def to_bfloat16_for_selected_params(self, precision: str) -> None:
            return None

    class FakePytorchModel:
        def __init__(self) -> None:
            self.paligemma_with_expert = FakePaliGemma()

    (tmp_path / "model.safetensors").touch()
    monkeypatch.setattr(pi0_config.Pi0Config, "load_pytorch", lambda *args, **kwargs: FakePytorchModel())
    monkeypatch.setattr(policy_module, "Policy", CapturingPolicy)

    policy_config.create_trained_policy(train_config, tmp_path, norm_stats={}, pytorch_device="cpu")

    assert captured["is_pytorch"] is True
    assert captured["metadata"] == {
        "existing": "value",
        "rtc_capabilities": {
            "algorithm": "disabled",
            "model_type": "pi05",
            "action_horizon": 4,
            "action_dim": 3,
        },
    }
    assert train_config.policy_metadata == original_metadata


def test_policy_passes_only_training_time_rtc_kwargs(monkeypatch) -> None:
    captured: dict = {}

    class FakeModel:
        def sample_actions(self, rng, observation, **kwargs):
            captured.update(kwargs)
            return policy_module.jnp.zeros((1, 4, 3), dtype=policy_module.jnp.float32)

    monkeypatch.setattr(policy_module.nnx_utils, "module_jit", lambda sample_actions: sample_actions)
    policy = policy_module.Policy(
        FakeModel(),
        metadata={"rtc_capabilities": make_capabilities(_train_config(enabled=True))},
        sample_kwargs={"rtc_target": "legacy", "rtc_weight": "legacy"},
    )
    observation = {"image": {}, "image_mask": {}, "state": np.zeros(3, dtype=np.float32)}

    policy.infer(observation, rtc=_training_time_request())

    assert set(captured) == {"rtc_action_prefix", "rtc_delay_steps"}
    assert captured["rtc_action_prefix"].shape == (1, 4, 3)
    assert captured["rtc_action_prefix"].dtype == policy_module.jnp.float32
    assert captured["rtc_delay_steps"].shape == (1,)
    assert captured["rtc_delay_steps"].dtype == policy_module.jnp.int32


def test_pytorch_policy_rejects_valid_training_time_rtc_request() -> None:
    class FakePytorchModel:
        def to(self, device):
            return self

        def eval(self) -> None:
            return None

        def sample_actions(self, device, observation, **kwargs):
            raise AssertionError("RTC request should be rejected before sampling")

    policy = policy_module.Policy(
        FakePytorchModel(),
        metadata={"rtc_capabilities": make_capabilities(_train_config(enabled=True))},
        is_pytorch=True,
    )
    observation = {"image": {}, "image_mask": {}, "state": np.zeros(3, dtype=np.float32)}

    with pytest.raises(RTCRequestError, match="PyTorch"):
        policy.infer(observation, rtc=_training_time_request())
