# Training-Time RTC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Add opt-in training-time RTC to JAX Pi0.5 and replace SpiritAI's experimental replacement-inpainting path with a YAML-configured asynchronous hard-prefix RTC runtime.

**Architecture:** Pi0.5 learns to denoise a postfix conditioned on a clean action prefix using per-token flow times. At deployment, a pure controller creates the prefix A[s:s+d] from the old plan; one background worker samples a new chunk while the main thread streams one safety-filtered action per control tick. No VJP, soft mask, beta, hybrid mode, or legacy-checkpoint fallback remains.

**Tech Stack:** Python 3.11, JAX, Flax NNX, NumPy, PyYAML, Tyro, WebSocket client/server, pytest, Ruff.

---

## File structure

| Path | Responsibility |
|---|---|
| src/openpi/training/config.py | RTC fine-tune config, validation, explicit h=50 configuration. |
| scripts/train.py | Pass static RTC loss settings only for enabled Pi0.5 training. |
| src/openpi/rtc/conditioning.py | Pure JAX prefix masks, training inputs and prefix freeze. |
| src/openpi/models/gemma.py | Global and tokenwise AdaRMS conditioning. |
| src/openpi/models/pi0.py | Tokenwise loss and hard-prefix Pi0.5 sampling. |
| src/openpi/rtc/capabilities.py | Checkpoint capability metadata and policy-request validation. |
| src/openpi/policies/policy.py | Host-side request validation and JAX sample kwargs. |
| src/openpi/policies/policy_config.py | Publish calculated RTC capability metadata. |
| src/openpi/rtc/timeline.py | Deterministic action-plan alignment, deadline and hold state. |
| src/openpi/rtc/runtime_config.py | Strict YAML schema and source-relative default path helper. |
| src/openpi/rtc/worker.py | One-inflight background policy worker. |
| examples/spirit-ai/main.py | Bootstrap config and the SpiritAI one-tick dispatcher. |
| examples/spirit-ai/configs/rtc/training_time.yaml | Checked-in runtime profile. |
| src/openpi/rtc/*_test.py | Unit tests for each pure RTC boundary. |
| examples/spirit-ai/main_test.py | Bootstrap and runtime adapter tests. |

Delete src/openpi/rtc/helpers.py and src/openpi/rtc/state.py only in the final migration task, after their replacements are tested.

### Task 1: Add static fine-tuning RTC config

**Files:**
- Create: src/openpi/training/config_test.py
- Modify: src/openpi/training/config.py
- Modify: scripts/train.py

- [ ] **Step 1: Write the failing configuration tests**

~~~python
# src/openpi/training/config_test.py
import pytest

from openpi.models import pi0_config
from openpi.training import config as training_config


def test_rtc_training_defaults_to_disabled() -> None:
    cfg = training_config.get_config("debug_pi05")
    assert cfg.rtc_training.enabled is False
    assert cfg.rtc_training.max_delay_steps == 0


def test_rtc_training_requires_pi05() -> None:
    with pytest.raises(ValueError, match="JAX Pi0.5"):
        training_config.TrainConfig(
            name="bad",
            exp_name="bad",
            model=pi0_config.Pi0Config(pi05=False),
            rtc_training=training_config.RTCTrainingConfig(enabled=True, max_delay_steps=4),
        )


def test_rtc_training_delay_must_fit_half_horizon() -> None:
    with pytest.raises(ValueError, match="floor\(action_horizon / 2\)"):
        training_config.TrainConfig(
            name="bad",
            exp_name="bad",
            model=pi0_config.Pi0Config(pi05=True, action_horizon=10),
            rtc_training=training_config.RTCTrainingConfig(enabled=True, max_delay_steps=6),
        )


def test_spiritai_h50_rtc_config_is_explicit() -> None:
    cfg = training_config.get_config("pi05_spiritai_cart_lora_h50_multiscale_rtc")
    assert cfg.model.pi05 is True
    assert cfg.model.action_horizon == 50
    assert cfg.rtc_training == training_config.RTCTrainingConfig(enabled=True, max_delay_steps=12)
~~~

- [ ] **Step 2: Run the test to verify it fails**

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest src/openpi/training/config_test.py -q

Expected: FAIL because RTCTrainingConfig and the h=50 RTC config do not exist.

- [ ] **Step 3: Add the config, its invariants, and static train-loop plumbing**

Add this frozen dataclass immediately before TrainConfig:

~~~python
@dataclasses.dataclass(frozen=True)
class RTCTrainingConfig:
    enabled: bool = False
    max_delay_steps: int = 0

    def __post_init__(self) -> None:
        if self.max_delay_steps < 0:
            raise ValueError(f"max_delay_steps must be non-negative, got {self.max_delay_steps}")
        if not self.enabled and self.max_delay_steps != 0:
            raise ValueError("max_delay_steps must be 0 when RTC training is disabled")
~~~

Add rtc_training with default_factory to TrainConfig. Extend its existing __post_init__:

~~~python
if self.rtc_training.enabled:
    if not isinstance(self.model, pi0_config.Pi0Config) or not self.model.pi05:
        raise ValueError("RTC training is supported only for JAX Pi0.5 models")
    if self.rtc_training.max_delay_steps > self.model.action_horizon // 2:
        raise ValueError(
            "RTC max_delay_steps must be <= floor(action_horizon / 2): "
            f"got {self.rtc_training.max_delay_steps} for H={self.model.action_horizon}"
        )
~~~

Register pi05_spiritai_cart_lora_h50_multiscale_rtc as a copy of the existing multiscale h=50 configuration, changing name, exp_name, and rtc_training to enabled with 12 steps. Keep 12 as the checked-in profile default; Task 9 defines how the operator changes it after real latency measurement.

In scripts/train.py, pass a keyword only when RTC is enabled:

~~~python
loss_kwargs: dict[str, int] = {}
if config.rtc_training.enabled:
    loss_kwargs["rtc_max_delay_steps"] = config.rtc_training.max_delay_steps
chunked_loss = model.compute_loss(rng, observation, actions, train=True, **loss_kwargs)
~~~

Add optional keyword-only rtc_max_delay_steps: int | None = None to Pi0.compute_loss only. Do not change BaseModel, Pi0-FAST, or PyTorch signatures.

- [ ] **Step 4: Verify and format**

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest src/openpi/training/config_test.py src/openpi/training/data_loader_test.py -q

Expected: PASS.

Run: uv run ruff check src/openpi/training/config.py scripts/train.py src/openpi/training/config_test.py

Expected: exit 0.

- [ ] **Step 5: Commit**

~~~bash
git add src/openpi/training/config.py src/openpi/training/config_test.py scripts/train.py
git commit -m "feat: add Pi0.5 RTC training config"
~~~

### Task 2: Add conditioning primitives and tokenwise AdaRMS

**Files:**
- Create: src/openpi/rtc/conditioning.py
- Create: src/openpi/rtc/conditioning_test.py
- Modify: src/openpi/models/gemma.py
- Modify: src/openpi/models/pi0.py
- Modify: src/openpi/models/model_test.py

- [ ] **Step 1: Write failing math and shape tests**

~~~python
# src/openpi/rtc/conditioning_test.py
import jax.numpy as jnp
import numpy as np

from openpi.rtc import conditioning


def test_prepare_training_inputs_keeps_prefix_clean() -> None:
    actions = jnp.arange(8, dtype=jnp.float32).reshape(2, 4, 1)
    noise = -jnp.ones_like(actions)
    time = jnp.array([0.25, 0.75], dtype=jnp.float32)
    delay = jnp.array([2, 0], dtype=jnp.int32)

    x_t, token_time, postfix = conditioning.prepare_training_inputs(actions, noise, time, delay)

    np.testing.assert_array_equal(np.asarray(x_t[0, :2]), np.asarray(actions[0, :2]))
    np.testing.assert_array_equal(np.asarray(token_time[0, :2]), np.zeros(2, dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(postfix[0]), np.array([False, False, True, True]))
    np.testing.assert_array_equal(np.asarray(postfix[1]), np.array([True, True, True, True]))


def test_masked_postfix_mean_weights_batch_rows_by_valid_tokens() -> None:
    loss = jnp.array([[3.0, 3.0, 2.0, 2.0], [4.0, 4.0, 4.0, 4.0]])
    postfix = jnp.array([[False, False, True, True], [True, True, True, True]])
    assert conditioning.masked_postfix_mean(loss, postfix) == 3.0
~~~

Add a model_test that creates Pi0Config with pi05=True, dummy PaliGemma/action expert and H=4. Call embed_suffix with timestep shape (1, 4) and assert that returned adarms_cond has shape (1, 4, action_expert_width).

- [ ] **Step 2: Run tests to verify they fail**

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest src/openpi/rtc/conditioning_test.py src/openpi/models/model_test.py -q

Expected: FAIL because the module and tokenwise timestep support do not exist.

- [ ] **Step 3: Implement pure JAX primitives**

~~~python
# src/openpi/rtc/conditioning.py
def prefix_mask(delay_steps: jax.Array, horizon: int) -> jax.Array:
    return jnp.arange(horizon)[None, :] < delay_steps[:, None]


def prepare_training_inputs(actions, noise, scalar_time, delay_steps):
    prefix = prefix_mask(delay_steps, actions.shape[-2])
    token_time = jnp.where(prefix, 0.0, scalar_time[:, None])
    x_t = jnp.where(
        prefix[..., None],
        actions,
        token_time[..., None] * noise + (1.0 - token_time[..., None]) * actions,
    )
    return x_t, token_time, jnp.logical_not(prefix)


def freeze_prefix(x_t, action_prefix, delay_steps):
    prefix = prefix_mask(delay_steps, x_t.shape[-2])
    return jnp.where(prefix[..., None], action_prefix, x_t)


def masked_postfix_mean(per_token_loss, postfix_mask):
    weights = postfix_mask.astype(per_token_loss.dtype)
    per_row = jnp.sum(per_token_loss * weights, axis=-1) / jnp.maximum(jnp.sum(weights, axis=-1), 1)
    return jnp.mean(per_row)
~~~

Use repository array typing annotations on public functions.

- [ ] **Step 4: Implement broadcast-safe AdaRMS and RTC loss**

In gemma.RMSNorm, replace unconditional modulation[:, None, :] with:

~~~python
modulation = nn.Dense(x.shape[-1] * 3, kernel_init=nn.initializers.zeros, dtype=dtype)(cond)
if modulation.ndim == 2:
    modulation = modulation[:, None, :]
scale, shift, gate = jnp.split(modulation, 3, axis=-1)
normed_inputs = normed_inputs * (1 + scale) + shift
return normed_inputs.astype(dtype), gate
~~~

In Pi0.embed_suffix, convert a global (B,) time into a (B,H) token_time and accept an already-tokenwise (B,H) input. For Pi0.5, feed token_time through the existing time MLP to get (B,H,E). For non-Pi0.5, use the same token_time per action token.

Leave Pi0.compute_loss bit-for-bit on its current path when rtc_max_delay_steps is None. In the RTC path split four RNGs, sample inclusive delays, and use the helper:

~~~python
delay = jax.random.randint(delay_rng, batch_shape, 0, rtc_max_delay_steps + 1)
x_t, token_time, postfix_mask = conditioning.prepare_training_inputs(actions, noise, scalar_time, delay)
per_token_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
postfix_count = jnp.sum(postfix_mask, axis=-1, keepdims=True)
return per_token_loss * postfix_mask * (self.action_horizon / postfix_count)
~~~

This preserves the current (B,H) return type while jnp.mean in scripts/train.py becomes the mean postfix loss per batch row.

- [ ] **Step 5: Verify and commit**

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest src/openpi/rtc/conditioning_test.py src/openpi/models/model_test.py -q

Expected: PASS.

Run: uv run ruff check src/openpi/rtc/conditioning.py src/openpi/rtc/conditioning_test.py src/openpi/models/gemma.py src/openpi/models/pi0.py src/openpi/models/model_test.py

Expected: exit 0.

~~~bash
git add src/openpi/rtc/conditioning.py src/openpi/rtc/conditioning_test.py src/openpi/models/gemma.py src/openpi/models/pi0.py src/openpi/models/model_test.py
git commit -m "feat: condition Pi0.5 on RTC action prefixes"
~~~

### Task 3: Add hard-prefix Pi0.5 sampling and remove replacement blending

**Files:**
- Modify: src/openpi/models/pi0.py
- Create: src/openpi/models/pi0_rtc_test.py

- [ ] **Step 1: Write failing sampler tests**

~~~python
import jax
import numpy as np
import pytest

from openpi.models import pi0_config


def test_pi05_sampling_returns_exact_prefix() -> None:
    cfg = pi0_config.Pi0Config(
        pi05=True, action_dim=32, action_horizon=4,
        paligemma_variant="dummy", action_expert_variant="dummy",
    )
    model = cfg.create(jax.random.key(0))
    prefix = np.full((1, 4, 32), 7.0, dtype=np.float32)
    actions = model.sample_actions(
        jax.random.key(1), cfg.fake_obs(batch_size=1), num_steps=2,
        rtc_action_prefix=prefix, rtc_delay_steps=np.array([2], dtype=np.int32),
    )
    np.testing.assert_array_equal(np.asarray(actions[:, :2]), prefix[:, :2])


def test_pi05_sampling_requires_prefix_and_delay_together() -> None:
    cfg = pi0_config.Pi0Config(pi05=True, action_horizon=4, paligemma_variant="dummy", action_expert_variant="dummy")
    model = cfg.create(jax.random.key(0))
    with pytest.raises(ValueError, match="must be provided together"):
        model.sample_actions(jax.random.key(1), cfg.fake_obs(), rtc_action_prefix=np.zeros((1, 4, 32), np.float32))
~~~

- [ ] **Step 2: Run test to verify it fails**

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest src/openpi/models/pi0_rtc_test.py -q

Expected: FAIL because the new sampler arguments are absent.

- [ ] **Step 3: Replace the sampler API**

Replace rtc_target, rtc_weight, rtc_beta and every replacement blend operation with:

~~~python
rtc_action_prefix: at.Float[at.Array, "b ah ad"] | None = None,
rtc_delay_steps: at.Int[at.Array, "b"] | None = None,
~~~

Before JIT loop, reject pair mismatches, non-Pi0.5 use, non-(B,H,D) prefix shape, non-(B,) delay shape, and delays outside 0 <= d < H. At each Euler iteration use:

~~~python
if rtc_action_prefix is not None:
    x_t = conditioning.freeze_prefix(x_t, rtc_action_prefix, rtc_delay_steps)
    token_time = jnp.where(
        conditioning.prefix_mask(rtc_delay_steps, self.action_horizon),
        0.0,
        jnp.broadcast_to(time, (batch_size, 1)),
    )
else:
    token_time = jnp.broadcast_to(time, (batch_size,))
suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, token_time)
~~~

After while_loop, call freeze_prefix again before returning. The no-RTC branch must not allocate a dummy target or weight.

- [ ] **Step 4: Verify**

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest src/openpi/models/pi0_rtc_test.py src/openpi/models/model_test.py -q

Expected: PASS.

Run: uv run ruff check src/openpi/models/pi0.py src/openpi/models/pi0_rtc_test.py

Expected: exit 0.

- [ ] **Step 5: Commit**

~~~bash
git add src/openpi/models/pi0.py src/openpi/models/pi0_rtc_test.py
git commit -m "feat: sample Pi0.5 RTC postfixes with hard prefixes"
~~~

### Task 4: Publish checkpoint capability metadata and validate policy RTC requests

**Files:**
- Create: src/openpi/rtc/capabilities.py
- Create: src/openpi/rtc/capabilities_test.py
- Modify: src/openpi/policies/policy.py
- Modify: src/openpi/policies/policy_config.py
- Modify: src/openpi/rtc/rtc_test.py

- [ ] **Step 1: Write failing capability tests**

~~~python
import numpy as np
import pytest

from openpi.rtc.capabilities import RTCRequestError
from openpi.rtc.capabilities import validate_training_time_request


def test_training_time_request_accepts_model_prefix() -> None:
    capability = {
        "algorithm": "training_time_v1", "action_horizon": 4, "action_dim": 32,
        "training_max_delay_steps": 2,
    }
    request = {
        "algorithm": "training_time_v1",
        "action_prefix": np.zeros((4, 32), np.float32),
        "delay_steps": 2,
    }
    prefix, delay = validate_training_time_request(request, capability)
    assert prefix.shape == (4, 32)
    assert delay == 2


def test_training_time_request_rejects_delay_over_capability() -> None:
    capability = {
        "algorithm": "training_time_v1", "action_horizon": 4, "action_dim": 32,
        "training_max_delay_steps": 2,
    }
    request = {
        "algorithm": "training_time_v1",
        "action_prefix": np.zeros((4, 32), np.float32),
        "delay_steps": 3,
    }
    with pytest.raises(RTCRequestError, match="training_max_delay_steps"):
        validate_training_time_request(request, capability)
~~~

Replace old target/mask/beta assertions in rtc_test.py with envelope assertions for algorithm, action_prefix, and delay_steps.

- [ ] **Step 2: Run test to verify it fails**

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest src/openpi/rtc/capabilities_test.py src/openpi/rtc/rtc_test.py -q

Expected: FAIL because capability functions do not exist.

- [ ] **Step 3: Implement metadata and strict request checks**

Create make_capabilities(train_config). For enabled Pi0.5 it returns:

~~~python
{
    "algorithm": "training_time_v1",
    "model_type": "pi05",
    "action_horizon": train_config.model.action_horizon,
    "action_dim": train_config.model.action_dim,
    "training_max_delay_steps": train_config.rtc_training.max_delay_steps,
}
~~~

For disabled config use algorithm disabled and omit training_max_delay_steps.

validate_training_time_request must require exactly algorithm, action_prefix, and delay_steps; reject disabled capability and unknown algorithm; convert prefix to float32; enforce (H,D); and enforce 0 <= delay <= training_max_delay_steps.

In create_trained_policy, use:

~~~python
metadata = dict(train_config.policy_metadata or {})
metadata.setdefault("rtc_capabilities", make_capabilities(train_config))
~~~

In Policy.infer, validate rtc before JIT. Convert returned prefix to JAX float32 (1,H,D) and delay to JAX int32 (1,). Pass only rtc_action_prefix and rtc_delay_steps. Remove target/mask/beta branches for both JAX and PyTorch.

- [ ] **Step 4: Verify**

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest src/openpi/rtc/capabilities_test.py src/openpi/rtc/rtc_test.py -q

Expected: PASS.

Run: uv run ruff check src/openpi/rtc/capabilities.py src/openpi/rtc/capabilities_test.py src/openpi/policies/policy.py src/openpi/policies/policy_config.py src/openpi/rtc/rtc_test.py

Expected: exit 0.

- [ ] **Step 5: Commit**

~~~bash
git add src/openpi/rtc/capabilities.py src/openpi/rtc/capabilities_test.py src/openpi/policies/policy.py src/openpi/policies/policy_config.py src/openpi/rtc/rtc_test.py
git commit -m "feat: validate training-time RTC policy requests"
~~~

### Task 5: Build the deterministic action-plan timeline

**Files:**
- Create: src/openpi/rtc/timeline.py
- Create: src/openpi/rtc/timeline_test.py
- Modify: src/openpi/rtc/__init__.py

- [ ] **Step 1: Write deterministic timeline tests**

~~~python
import numpy as np

from openpi.rtc.timeline import ActionPlan
from openpi.rtc.timeline import RTCController


def make_plan(generation_tick: int, base: float) -> ActionPlan:
    model = np.arange(8, dtype=np.float32).reshape(8, 1) + base
    return ActionPlan(generation_tick=generation_tick, model_actions=model, robot_actions=model.copy())


def test_request_prefix_is_old_plan_shifted_by_execution_horizon() -> None:
    controller = RTCController(action_horizon=8, action_dim=1, s_min=2, training_max_delay_steps=3)
    controller.install_initial_plan(make_plan(0, 0.0))
    request = controller.start_request(current_tick=3, planned_delay_steps=3)
    np.testing.assert_array_equal(request.action_prefix[:3], np.array([[3.0], [4.0], [5.0]], dtype=np.float32))
    assert request.execution_horizon == 3


def test_late_result_is_rejected_without_switching() -> None:
    controller = RTCController(action_horizon=8, action_dim=1, s_min=2, training_max_delay_steps=3)
    old = make_plan(0, 0.0)
    controller.install_initial_plan(old)
    request = controller.start_request(current_tick=3, planned_delay_steps=2)
    assert controller.accept_result(request, make_plan(3, 100.0), completion_tick=6) is False
    np.testing.assert_array_equal(controller.action_for_tick(6).model_action, old.model_actions[6])
~~~

Add tests for d=0, s_min > d, d > H-s, one in-flight request, result at d_plan switching after the frozen prefix, exhausted plan returning hold, and rejected robot ack not advancing the tick.

- [ ] **Step 2: Run test to verify it fails**

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest src/openpi/rtc/timeline_test.py -q

Expected: FAIL because ActionPlan and RTCController do not exist.

- [ ] **Step 3: Implement pure controller state transitions**

Implement ActionPlan, RTCRequest, DispatchAction and RTCController. ActionPlan validates model shape (H,D), matching robot horizon, and stores generation_tick.

start_request must use exactly:

~~~python
s = max(planned_delay_steps, self.s_min)
if current_tick != self._active.generation_tick + s:
    raise RTCStateError("RTC request must start exactly at the execution horizon")
if planned_delay_steps > self.training_max_delay_steps:
    raise RTCStateError("planned delay exceeds training capability")
if planned_delay_steps > self.action_horizon - s:
    raise RTCStateError("planned delay violates d <= H - s")
prefix = self._active.model_actions[s : s + planned_delay_steps]
~~~

accept_result computes actual_delay = completion_tick - request.start_tick. Reject and record a deadline miss when actual_delay is greater than planned_delay_steps. Otherwise install the returned plan at request.start_tick. action_for_tick indexes tick - generation_tick; it returns DispatchAction(kind="hold", model_action=None, robot_action=None) when no action is available. record_accepted_tick is the only method that increments the tick.

- [ ] **Step 4: Verify**

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest src/openpi/rtc/timeline_test.py -q

Expected: PASS.

Run: uv run ruff check src/openpi/rtc/timeline.py src/openpi/rtc/timeline_test.py src/openpi/rtc/__init__.py

Expected: exit 0.

- [ ] **Step 5: Commit**

~~~bash
git add src/openpi/rtc/timeline.py src/openpi/rtc/timeline_test.py src/openpi/rtc/__init__.py
git commit -m "feat: add deterministic RTC action timeline"
~~~

### Task 6: Add strict YAML parsing and the default profile

**Files:**
- Modify: pyproject.toml
- Modify: uv.lock
- Create: src/openpi/rtc/runtime_config.py
- Create: src/openpi/rtc/runtime_config_test.py
- Create: examples/spirit-ai/configs/rtc/training_time.yaml

- [ ] **Step 1: Write failing parser tests**

~~~python
from pathlib import Path

import pytest
import yaml

from openpi.rtc.runtime_config import RuntimeConfigError
from openpi.rtc.runtime_config import default_config_path
from openpi.rtc.runtime_config import load_runtime_config


VALID_RUNTIME = {
    "schema_version": 1,
    "policy": {"host": "localhost", "port": 8000, "prompt": "fold"},
    "robot": {
        "url": "ws://robot", "action_layout": "cartesian", "enable_external_following": False,
        "initial_gripper_obs_state": 0.0965, "gripper_initial_tolerance": 0.00965,
        "gripper_reset_command_state": 1.0, "gripper_reset_steps": 10,
    },
    "control": {
        "source_hz": 15.0, "max_steps": 20, "busy_sleep_s": 0.01, "startup_delay_s": 0.0,
        "blend_steps": 4, "rollback_guard_steps": 4, "rollback_scale": 0.2,
        "rpc_budget_fraction": 0.7,
        "motion_limits": {
            "max_arm_velocity_rad_s": 0.35, "max_torso_velocity_rad_s": 0.2,
            "max_gripper_velocity_s": 0.8, "max_base_speed": 0.05,
            "max_joint_accel_rad_s2": 0.0, "max_cart_translation_m_s": 0.08,
            "max_cart_rotation_rad_s": 0.35, "max_torso_cart_translation_m_s": 0.04,
            "max_torso_cart_rotation_rad_s": 0.2, "max_cart_accel": 0.0,
        },
    },
    "rtc": {
        "mode": "training_time", "s_min": 5,
        "delay": {"planned_max_steps": 12, "history_window": 16, "safety_margin_steps": 1},
        "deadline_miss": {"max_consecutive": 2, "action": "hold_then_stop"},
    },
}


def test_default_config_path_is_entrypoint_relative(tmp_path: Path) -> None:
    assert default_config_path(tmp_path / "main.py") == tmp_path / "configs/rtc/training_time.yaml"


def test_parser_rejects_unknown_top_level_key(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("schema_version: 1\nunknown: true\n")
    with pytest.raises(RuntimeConfigError, match="unknown keys"):
        load_runtime_config(path)


def test_parser_loads_training_time_profile(tmp_path: Path) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text(yaml.safe_dump(VALID_RUNTIME))
    cfg = load_runtime_config(path)
    assert cfg.rtc.mode == "training_time"
    assert cfg.control.source_hz == 15.0
~~~

- [ ] **Step 2: Run test to verify it fails**

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest src/openpi/rtc/runtime_config_test.py -q

Expected: FAIL because runtime_config.py is absent.

- [ ] **Step 3: Implement strict config schema**

Add pyyaml>=6.0 to project dependencies and run uv lock.

Create frozen dataclasses PolicyRuntimeConfig, RobotRuntimeConfig, ControlRuntimeConfig, MotionLimitsRuntimeConfig, RTCDelayRuntimeConfig, RTCDeadlineRuntimeConfig, RTCRuntimeConfig, and RuntimeConfig. RobotRuntimeConfig must hold url, action_layout, enable_external_following, initial_gripper_obs_state, gripper_initial_tolerance, gripper_reset_command_state and gripper_reset_steps. ControlRuntimeConfig must hold source_hz, max_steps, busy_sleep_s, startup_delay_s, blend_steps, rollback_guard_steps, rollback_scale, rpc_budget_fraction and MotionLimitsRuntimeConfig. Implement a helper that rejects set(mapping) - expected_keys. Use yaml.safe_load and require exactly schema_version, policy, robot, control, and rtc at top level.

Enforce:

~~~python
if config.rtc.mode != "training_time":
    raise RuntimeConfigError("rtc.mode must be 'training_time'")
if config.control.source_hz <= 0:
    raise RuntimeConfigError("control.source_hz must be positive")
if not 0 < config.control.rpc_budget_fraction < 1:
    raise RuntimeConfigError("control.rpc_budget_fraction must be in (0, 1)")
if config.rtc.s_min < 0 or config.rtc.delay.planned_max_steps < 0:
    raise RuntimeConfigError("RTC step counts must be non-negative")
if config.rtc.deadline_miss.action != "hold_then_stop":
    raise RuntimeConfigError("deadline_miss.action must be 'hold_then_stop'")
~~~

default_config_path(entrypoint) must return entrypoint.parent / "configs" / "rtc" / "training_time.yaml" and never read a file.

- [ ] **Step 4: Add default YAML and verify**

Create the default YAML with every former runtime default encoded explicitly:

~~~yaml
schema_version: 1
policy: {host: localhost, port: 8000, prompt: fold the paper box}
robot:
  url: ws://172.16.0.30:8766
  action_layout: cartesian
  enable_external_following: false
  initial_gripper_obs_state: 0.0965
  gripper_initial_tolerance: 0.00965
  gripper_reset_command_state: 1.0
  gripper_reset_steps: 10
control:
  source_hz: 15.0
  max_steps: 2000
  busy_sleep_s: 0.01
  startup_delay_s: 10.0
  blend_steps: 4
  rollback_guard_steps: 4
  rollback_scale: 0.2
  rpc_budget_fraction: 0.7
  motion_limits:
    max_arm_velocity_rad_s: 0.35
    max_torso_velocity_rad_s: 0.2
    max_gripper_velocity_s: 0.8
    max_base_speed: 0.05
    max_joint_accel_rad_s2: 0.0
    max_cart_translation_m_s: 0.08
    max_cart_rotation_rad_s: 0.35
    max_torso_cart_translation_m_s: 0.04
    max_torso_cart_rotation_rad_s: 0.2
    max_cart_accel: 0.0
rtc:
  mode: training_time
  s_min: 5
  delay: {planned_max_steps: 12, history_window: 16, safety_margin_steps: 1}
  deadline_miss: {max_consecutive: 2, action: hold_then_stop}
~~~

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest src/openpi/rtc/runtime_config_test.py -q

Expected: PASS.

Run: uv run ruff check src/openpi/rtc/runtime_config.py src/openpi/rtc/runtime_config_test.py

Expected: exit 0.

- [ ] **Step 5: Commit**

~~~bash
git add pyproject.toml uv.lock src/openpi/rtc/runtime_config.py src/openpi/rtc/runtime_config_test.py examples/spirit-ai/configs/rtc/training_time.yaml
git commit -m "feat: add YAML RTC runtime configuration"
~~~

### Task 7: Add the one-inflight background worker

**Files:**
- Create: src/openpi/rtc/worker.py
- Create: src/openpi/rtc/worker_test.py

- [ ] **Step 1: Write failing worker tests**

~~~python
import threading
import time

import pytest

from openpi.rtc.worker import RTCInferenceWorker
from openpi.rtc.worker import RTCWorkerBusyError


def test_worker_returns_value_and_monotonic_times() -> None:
    worker = RTCInferenceWorker(lambda request: request * 2)
    result = worker.submit(3).result(timeout=1)
    assert result.value == 6
    assert result.started_at <= result.finished_at <= time.monotonic()
    worker.close()


def test_worker_rejects_second_inflight_request() -> None:
    started = threading.Event()
    release = threading.Event()

    def wait_for_release(request: str) -> str:
        started.set()
        assert release.wait(timeout=1)
        return request

    worker = RTCInferenceWorker(wait_for_release)
    worker.submit("first")
    assert started.wait(timeout=1)
    with pytest.raises(RTCWorkerBusyError):
        worker.submit("second")
    release.set()
    worker.close()
~~~

- [ ] **Step 2: Run test to verify it fails**

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest src/openpi/rtc/worker_test.py -q

Expected: FAIL because RTCInferenceWorker is absent.

- [ ] **Step 3: Implement the worker**

Use ThreadPoolExecutor(max_workers=1), RTCInferenceResult(value, started_at, finished_at), and a stored Future:

~~~python
T = typing.TypeVar("T")


@dataclasses.dataclass(frozen=True)
class RTCInferenceResult(typing.Generic[T]):
    value: T
    started_at: float
    finished_at: float


class RTCInferenceWorker(typing.Generic[T]):
    def submit(self, request: T) -> Future[RTCInferenceResult[T]]:
        if self._future is not None and not self._future.done():
            raise RTCWorkerBusyError("RTC inference request already in flight")
        self._future = self._executor.submit(self._run, request)
        return self._future

    def _run(self, request: T) -> RTCInferenceResult[T]:
        started_at = time.monotonic()
        value = self._infer(request)
        return RTCInferenceResult(value=value, started_at=started_at, finished_at=time.monotonic())
~~~

close calls shutdown(wait=True, cancel_futures=False).

The worker owns only the policy client. No robot websocket object may be passed into worker request data.

- [ ] **Step 4: Verify**

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest src/openpi/rtc/worker_test.py -q

Expected: PASS.

Run: uv run ruff check src/openpi/rtc/worker.py src/openpi/rtc/worker_test.py

Expected: exit 0.

- [ ] **Step 5: Commit**

~~~bash
git add src/openpi/rtc/worker.py src/openpi/rtc/worker_test.py
git commit -m "feat: add single-flight RTC inference worker"
~~~

### Task 8: Replace SpiritAI's synchronous RTC loop

**Files:**
- Modify: examples/spirit-ai/main.py
- Create: examples/spirit-ai/main_test.py
- Modify: src/openpi/rtc/timeline.py
- Modify: src/openpi/rtc/timeline_test.py

- [ ] **Step 1: Write adapter tests**

Load examples/spirit-ai/main.py with importlib.util.spec_from_file_location. Test:

~~~python
def test_default_config_is_source_relative() -> None:
    assert module.DEFAULT_RUNTIME_CONFIG == Path(module.__file__).parent / "configs/rtc/training_time.yaml"


def test_metadata_rejects_delay_over_checkpoint_limit() -> None:
    runtime = load_runtime_config(module.DEFAULT_RUNTIME_CONFIG)
    metadata = {
        "rtc_capabilities": {
            "algorithm": "training_time_v1",
            "action_horizon": 50,
            "action_dim": 32,
            "training_max_delay_steps": 8,
        }
    }
    with pytest.raises(ValueError, match="planned_max_steps"):
        module.validate_rtc_runtime_metadata(runtime, metadata)
~~~

Add timeline assertion that a rejected robot command ack does not advance its logical tick.

- [ ] **Step 2: Run test to verify it fails**

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest examples/spirit-ai/main_test.py src/openpi/rtc/timeline_test.py -q

Expected: FAIL because bootstrap config and metadata validator are absent.

- [ ] **Step 3: Implement bootstrap and runtime validation**

At module scope define:

~~~python
DEFAULT_RUNTIME_CONFIG = Path(__file__).parent / "configs" / "rtc" / "training_time.yaml"


@dataclasses.dataclass(frozen=True)
class BootstrapArgs:
    config: Path = DEFAULT_RUNTIME_CONFIG
    dry_run: bool = False
~~~

Replace the old large Tyro Args interface with BootstrapArgs. Load YAML only inside main after Tyro parsing. Log resolved config path, robot URL, layout, RTC mode, and source Hz.

Read policy metadata once, require algorithm training_time_v1, and reject planned_max_steps over training_max_delay_steps. Derive model H and D only from metadata; never accept manual model dimension CLI fields.

- [ ] **Step 4: Implement exact one-tick dispatch order**

Delete prefetched_chunk, RTCState, enable_rtc, rtc_beta, rtc_s_min, action-dimension flags and all post-ack prefetch code. Keep reset, observation mapping and current safety helpers.

Use this loop:

~~~text
1. Wait for idle, then get robot observation and current state.
2. Poll one worker future and give a finished result to controller.accept_result(current_tick).
3. If current_tick equals active generation_tick + execution_horizon, start one worker request with prefix A[s:s+d].
4. Get controller.action_for_tick(current_tick); create a one-step safe hold when kind is hold.
5. Run rollback suppression, blend and motion limits on a (1, command_dim) array.
6. In non-dry-run mode send command and require accepted ack; only then record_accepted_tick.
7. Record read-only preflight, command ack and inference durations; stop after configured consecutive misses.
~~~

Worker inference calls _infer_policy_chunk with:

~~~python
rtc = {
    "algorithm": "training_time_v1",
    "action_prefix": request.action_prefix,
    "delay_steps": request.planned_delay_steps,
}
~~~

It must request return_model_actions=True, create ActionPlan from returned raw 32D model actions plus robot actions, and retain request.start_tick as the new plan logical generation tick. The main thread must never call policy.infer while worker exists.

The preflight only calls get_status and get_obs. Never send an action merely to benchmark the robot. Compare read-only RPC duration plus rolling accepted-command ack duration to (1 / source_hz) * rpc_budget_fraction; a violation prevents a request/retry, never accepts an early plan.

- [ ] **Step 5: Verify and commit**

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest examples/spirit-ai/main_test.py src/openpi/rtc/timeline_test.py src/openpi/policies/spiritai_bridge_test.py -q

Expected: PASS.

Run: uv run ruff check examples/spirit-ai/main.py examples/spirit-ai/main_test.py

Expected: exit 0.

~~~bash
git add examples/spirit-ai/main.py examples/spirit-ai/main_test.py src/openpi/rtc/timeline.py src/openpi/rtc/timeline_test.py
git commit -m "feat: run SpiritAI with asynchronous training-time RTC"
~~~

### Task 9: Remove replacement artifacts, document, and verify end-to-end

**Files:**
- Delete: src/openpi/rtc/helpers.py
- Delete: src/openpi/rtc/state.py
- Modify: src/openpi/rtc/rtc_test.py
- Modify: src/openpi/rtc/__init__.py
- Modify: examples/spirit-ai/README.md
- Modify: docs/superpowers/specs/2026-08-07-training-time-rtc-design.md

- [ ] **Step 1: Replace legacy API tests**

~~~python
import inspect

from openpi.models.pi0 import Pi0


def test_pi0_exposes_only_training_time_rtc_arguments() -> None:
    params = inspect.signature(Pi0.sample_actions).parameters
    assert "rtc_target" not in params
    assert "rtc_weight" not in params
    assert "rtc_beta" not in params
    assert "rtc_action_prefix" in params
    assert "rtc_delay_steps" in params
~~~

- [ ] **Step 2: Run migration test before removal**

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest src/openpi/rtc/rtc_test.py -q

Expected: PASS after Tasks 3 through 5; only then delete helpers.py and state.py and their imports.

- [ ] **Step 3: Update operator documentation**

Replace the existing SpiritAI RTC section with these exact operational facts:

~~~text
- RTC requires a JAX Pi0.5 checkpoint trained with rtc_training.enabled.
- Sampling uses hard action-prefix conditioning; VJP, beta, soft mask and legacy mode are unsupported.
- uv run examples/spirit-ai/main.py uses the source-relative default YAML.
- --config PATH chooses another strict YAML profile; --dry-run suppresses command sends.
- Measure latency before training; training_max_delay_steps and planned_max_steps must agree.
~~~

Add required run metrics: d_plan, d_actual, deadline misses, holds, command delta, control frequency and end-to-end inference latency. Change design-doc status to approved, pending implementation and link this plan.

- [ ] **Step 4: Run non-hardware verification**

~~~bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest   src/openpi/training/config_test.py   src/openpi/rtc/conditioning_test.py   src/openpi/models/pi0_rtc_test.py   src/openpi/rtc/capabilities_test.py   src/openpi/rtc/timeline_test.py   src/openpi/rtc/runtime_config_test.py   src/openpi/rtc/worker_test.py   src/openpi/rtc/rtc_test.py   src/openpi/models/model_test.py   src/openpi/policies/spiritai_policy_test.py   src/openpi/policies/spiritai_bridge_test.py   examples/spirit-ai/main_test.py -q
uv run ruff check src/openpi/rtc src/openpi/models/pi0.py src/openpi/models/gemma.py src/openpi/policies/policy.py src/openpi/policies/policy_config.py src/openpi/training/config.py scripts/train.py examples/spirit-ai/main.py
git diff --check
~~~

Expected: all pytest targets pass, Ruff exits 0, and git diff --check has no output.

Before physical motion, run default profile with --dry-run, verify the resolved configuration and capability metadata, then perform only read-only robot RPC preflight. Fine-tune and low-speed hardware smoke testing begin only after the operator chooses measured delay values.

- [ ] **Step 5: Commit**

~~~bash
git add -A
git commit -m "feat: replace SpiritAI RTC with training-time conditioning"
git status --short --branch
~~~

Expected: intended branch and no uncommitted files.

## Plan self-review

**Spec coverage:** Task 1 adds a default-off training control and explicit h=50 fine-tune configuration. Tasks 2 and 3 implement per-token time, clean prefix, postfix-only loss and hard freeze. Task 4 defines the model-space WebSocket contract and checkpoint metadata. Tasks 5, 7 and 8 implement conservative delay scheduling, A[s:s+d] alignment, single-action execution and safe hold. Task 6 implements source-relative default YAML and --config override. Task 9 removes replacement inpainting and covers documentation and verification.

**Placeholder scan:** Every task names exact files, commands, expected results, validation rules and commit boundaries. The checked-in 12-step default is explicit; deployment can change it only through the documented latency measurement process.

**Type consistency:** Model sampling uses rtc_action_prefix (B,H,D) and rtc_delay_steps (B,). WebSocket uses action_prefix (H,D) and scalar delay_steps. Metadata uses training_max_delay_steps. YAML uses planned_max_steps. Timeline uses planned_delay_steps and calculates actual_delay from control ticks.
