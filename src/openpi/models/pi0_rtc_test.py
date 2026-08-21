import jax
import jax.numpy as jnp
import pytest

from openpi.models import pi0_config
from openpi.shared import nnx_utils


@pytest.fixture(scope="module")
def pi05_model():
    config = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=4,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    return config.create(jax.random.key(0)), config


@pytest.fixture(scope="module")
def pi0_model():
    config = pi0_config.Pi0Config(
        action_horizon=4,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    return config.create(jax.random.key(1)), config


def _rtc_inputs(config):
    action_prefix = jnp.arange(config.action_horizon * config.action_dim, dtype=jnp.float32).reshape(
        1, config.action_horizon, config.action_dim
    )
    return action_prefix, jnp.array([2], dtype=jnp.int32)


def test_sample_actions_keeps_rtc_prefix_exact(pi05_model):
    model, config = pi05_model
    observation = config.fake_obs()
    action_prefix, delay_steps = _rtc_inputs(config)

    actions = model.sample_actions(
        jax.random.key(2),
        observation,
        num_steps=2,
        noise=jnp.zeros_like(action_prefix),
        rtc_action_prefix=action_prefix,
        rtc_delay_steps=delay_steps,
    )

    assert jnp.array_equal(actions[:, :2], action_prefix[:, :2])


@pytest.mark.parametrize("include_prefix", [False, True])
def test_sample_actions_requires_rtc_arguments_together(pi05_model, include_prefix):
    model, config = pi05_model
    action_prefix, delay_steps = _rtc_inputs(config)
    kwargs = {"rtc_action_prefix": action_prefix} if include_prefix else {"rtc_delay_steps": delay_steps}

    with pytest.raises(ValueError, match="must be provided together"):
        model.sample_actions(jax.random.key(3), config.fake_obs(), num_steps=1, **kwargs)


def test_sample_actions_rejects_rtc_for_pi0(pi0_model):
    model, config = pi0_model
    action_prefix, delay_steps = _rtc_inputs(config)

    with pytest.raises(ValueError, match="Pi0.5"):
        model.sample_actions(
            jax.random.key(4),
            config.fake_obs(),
            num_steps=1,
            rtc_action_prefix=action_prefix,
            rtc_delay_steps=delay_steps,
        )


@pytest.mark.parametrize(
    ("action_prefix", "delay_steps", "match"),
    [
        (jnp.zeros((1, 3, 32), dtype=jnp.float32), jnp.array([2], dtype=jnp.int32), "rtc_action_prefix"),
        (jnp.zeros((1, 4, 32), dtype=jnp.float32), jnp.array([[2]], dtype=jnp.int32), "rtc_delay_steps"),
    ],
)
def test_sample_actions_validates_rtc_input_shapes(pi05_model, action_prefix, delay_steps, match):
    model, config = pi05_model

    with pytest.raises(ValueError, match=match):
        model.sample_actions(
            jax.random.key(5),
            config.fake_obs(),
            num_steps=1,
            rtc_action_prefix=action_prefix,
            rtc_delay_steps=delay_steps,
        )


@pytest.mark.parametrize("delay_steps", [jnp.array([-1], dtype=jnp.int32), jnp.array([4], dtype=jnp.int32)])
def test_sample_actions_rejects_eager_out_of_range_rtc_delay(pi05_model, delay_steps):
    model, config = pi05_model
    action_prefix, _ = _rtc_inputs(config)

    with pytest.raises(ValueError, match="rtc_delay_steps"):
        model.sample_actions(
            jax.random.key(6),
            config.fake_obs(),
            num_steps=1,
            rtc_action_prefix=action_prefix,
            rtc_delay_steps=delay_steps,
        )


def test_sample_actions_rejects_jitted_out_of_range_rtc_delay(pi05_model):
    model, config = pi05_model
    action_prefix, _ = _rtc_inputs(config)
    sample_actions = nnx_utils.module_jit(model.sample_actions)

    def sample_invalid_delay():
        actions = sample_actions(
            jax.random.key(7),
            config.fake_obs(),
            num_steps=1,
            rtc_action_prefix=action_prefix,
            rtc_delay_steps=jnp.array([config.action_horizon], dtype=jnp.int32),
        )
        return jax.block_until_ready(actions)

    with pytest.raises(Exception, match="rtc_delay_steps must satisfy"):
        sample_invalid_delay()
