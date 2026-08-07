import jax.numpy as jnp

from openpi.rtc import conditioning


def test_prepare_training_inputs_preserves_clean_prefix():
    actions = jnp.arange(8, dtype=jnp.float32).reshape(2, 4, 1)
    noise = -jnp.ones_like(actions)
    scalar_time = jnp.array([0.25, 0.75], dtype=jnp.float32)
    delay_steps = jnp.array([2, 0], dtype=jnp.int32)

    x_t, token_time, postfix_mask = conditioning.prepare_training_inputs(
        actions, noise, scalar_time, delay_steps
    )

    assert jnp.array_equal(x_t[0, :2], actions[0, :2])
    assert jnp.array_equal(token_time[0, :2], jnp.zeros(2, dtype=jnp.float32))
    assert jnp.array_equal(postfix_mask[0], jnp.array([False, False, True, True]))
    assert jnp.array_equal(postfix_mask[1], jnp.array([True, True, True, True]))


def test_masked_postfix_mean_equally_weights_postfix_means():
    loss = jnp.array([[3.0, 3.0, 2.0, 2.0], [4.0, 4.0, 4.0, 4.0]])
    postfix_mask = jnp.array([[False, False, True, True], [True, True, True, True]])

    assert conditioning.masked_postfix_mean(loss, postfix_mask) == 3.0
