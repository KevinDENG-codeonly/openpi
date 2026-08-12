import pytest

from openpi.models import pi0_config
from openpi.training import config as _config


def test_debug_pi05_rtc_training_is_disabled_by_default():
    config = _config.get_config("debug_pi05")

    assert config.rtc_training.enabled is False
    assert config.rtc_training.max_delay_steps == 0


def test_rtc_training_requires_jax_pi05():
    with pytest.raises(ValueError, match="JAX Pi0.5"):
        _config.TrainConfig(
            name="bad",
            exp_name="bad",
            model=pi0_config.Pi0Config(pi05=False),
            rtc_training=_config.RTCTrainingConfig(enabled=True, max_delay_steps=4),
        )


def test_rtc_training_delay_must_not_exceed_half_action_horizon():
    with pytest.raises(ValueError, match=r"floor\(action_horizon / 2\)"):
        _config.TrainConfig(
            name="bad",
            exp_name="bad",
            model=pi0_config.Pi0Config(pi05=True, action_horizon=10),
            rtc_training=_config.RTCTrainingConfig(enabled=True, max_delay_steps=6),
        )


def test_pi05_spiritai_cart_lora_h50_multiscale_rtc_config():
    config = _config.get_config("pi05_spiritai_cart_lora_h50_multiscale_rtc")

    assert config.model.pi05 is True
    assert config.model.action_horizon == 50
    assert config.rtc_training == _config.RTCTrainingConfig(enabled=True, max_delay_steps=12)


def test_pi05_spiritai_cart_lora_h50_20260805_14annotations_rtc_config():
    config = _config.get_config("pi05_spiritai_cart_lora_h50_20260805_14annotations")

    assert config.model.pi05 is True
    assert config.model.action_horizon == 50
    assert config.rtc_training == _config.RTCTrainingConfig(enabled=True, max_delay_steps=20)
