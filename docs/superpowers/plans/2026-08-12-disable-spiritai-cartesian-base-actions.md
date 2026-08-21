# Disable SpiritAI Cartesian Base Actions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent malformed base-speed labels from destabilizing the target SpiritAI Cartesian LoRA training run, while guaranteeing that deployment never sends a base command.

**Architecture:** Keep the 25-dimensional Cartesian action interface unchanged. Add an opt-in flag to the existing Cartesian input transform that overwrites only the final three base-speed training targets with zero before normalization; enable it only in the 14-annotations training config. Set the existing deployment-side base-speed safety limit to zero so a 23-dimensional robot-server command is still safe even if a policy returns base values.

**Tech Stack:** Python 3.11, NumPy, pytest, OpenPI data transforms, YAML runtime configuration.

---

### Task 1: Add a scoped base-target suppression switch

**Files:**
- Modify: `src/openpi/policies/spiritai_policy.py:172-182`
- Modify: `src/openpi/training/config.py:413-450,1123-1127`
- Test: `src/openpi/policies/spiritai_policy_test.py:43-60`
- Test: `src/openpi/training/config_test.py:35-42`

- [x] **Step 1: Write failing transform and config assertions**

```python
transform = spiritai_policy.SpiritaiCartesianInputs(
    model_type=_model.ModelType.PI05,
    zero_base_action_targets=True,
)
out = transform(data)

np.testing.assert_array_equal(out["actions"][:, -3:], np.zeros((2, 3), dtype=np.float32))
assert _config.get_config("pi05_spiritai_cart_lora_h50_20260805_14annotations").data.zero_base_action_targets
```

- [x] **Step 2: Run the focused tests and confirm they fail because the new argument and config field do not exist**

Run:

```bash
uv run pytest src/openpi/policies/spiritai_policy_test.py src/openpi/training/config_test.py -q
```

Expected: failure mentioning `zero_base_action_targets`.

- [x] **Step 3: Implement the smallest scoped behavior**

```python
@dataclasses.dataclass(frozen=True)
class SpiritaiCartesianInputs(transforms.DataTransformFn):
    model_type: _model.ModelType
    zero_base_action_targets: bool = False

    def __call__(self, data: dict) -> dict:
        inputs = _build_policy_inputs(
            data,
            state_keys=CARTESIAN_STATE_KEYS,
            action_keys=CARTESIAN_ACTION_KEYS,
        )
        if self.zero_base_action_targets and "actions" in inputs:
            inputs["actions"][..., -3:] = 0.0
        return inputs
```

Add `zero_base_action_targets: bool = False` to `LeRobotSpiritaiCartesianDataConfig`, pass it to `SpiritaiCartesianInputs`, and set it to `True` only in `pi05_spiritai_cart_lora_h50_20260805_14annotations`.

- [x] **Step 4: Run the focused tests and confirm they pass**

Run:

```bash
uv run pytest src/openpi/policies/spiritai_policy_test.py src/openpi/training/config_test.py -q
```

Expected: PASS.

### Task 2: Make the checked-in deployment profile hard-disable base motion

**Files:**
- Modify: `examples/spirit-ai/configs/rtc/training_time.yaml:28`
- Test: `src/openpi/rtc/runtime_config_test.py:80-94`

- [x] **Step 1: Write a failing checked-in-profile assertion**

```python
config = load_runtime_config(profile_path)
assert config.control.motion_limits.max_base_speed == 0.0
```

- [x] **Step 2: Set the profile limit**

```yaml
# This folding policy never controls the mobile base.
max_base_speed: 0.0
```

The existing bridge test already proves that the safety limiter clamps the three base-speed fields for a 23-dimensional Cartesian command; no bridge implementation change is required.

- [x] **Step 3: Run the bridge and runtime-profile checks**

Run:

```bash
uv run pytest src/openpi/policies/spiritai_bridge_test.py src/openpi/rtc/runtime_config_test.py -q
uv run python -c 'from pathlib import Path; from openpi.rtc.runtime_config import load_runtime_config; cfg = load_runtime_config(Path("examples/spirit-ai/configs/rtc/training_time.yaml")); assert cfg.control.motion_limits.max_base_speed == 0.0'
```

Expected: PASS with no output from the assertion command.

### Task 3: Rebuild and validate compatible normalization statistics

**Files:**
- Modify (generated): `assets/pi05_spiritai_cart_lora_h50_20260805_14annotations/spiritai/20260805_FoldBox_SpiritAI_Moz1WB_14Annotations/norm_stats.json`

- [x] **Step 1: Rebuild the only affected statistics exactly**

`zero_base_action_targets` changes only the final three `actions` coordinates and changes every value in those coordinates to zero. The existing stats were generated after the current local dataset was available, so state and the first 22 action coordinates remain valid. Verify the same `RunningStats` implementation yields zero for an all-zero action block, then update only the three final values of `mean`, `std`, `q01`, and `q99` to `0.0`.

```bash
uv run python -c 'import numpy as np; from openpi.shared.normalize import RunningStats; s = RunningStats(); s.update(np.zeros((32, 50, 3), dtype=np.float32)); r = s.get_statistics(); assert all(np.all(v == 0.0) for v in (r.mean, r.std, r.q01, r.q99))'
```

Expected: no output. This is mathematically equivalent to a full rescan for the only transformed dimensions and avoids unnecessary image decoding for unchanged dimensions.

- [x] **Step 2: Verify the final three action dimensions are all zero in saved stats**

Run:

```bash
uv run python -c 'import json; from pathlib import Path; p = Path("assets/pi05_spiritai_cart_lora_h50_20260805_14annotations/spiritai/20260805_FoldBox_SpiritAI_Moz1WB_14Annotations/norm_stats.json"); s = json.loads(p.read_text())["actions"]; assert all(s[k][-3:] == [0.0, 0.0, 0.0] for k in ("mean", "q01", "q99")), {k: s[k][-3:] for k in ("mean", "q01", "q99")}'
```

Expected: no output; action base-speed zero normalizes to a stable constant instead of producing quantile outliers.

### Task 4: Run a no-checkpoint training preflight

**Files:**
- No source changes.

- [x] **Step 1: Start a 101-step run with the target config**

Run:

```bash
uv run python scripts/train.py pi05_spiritai_cart_lora_h50_20260805_14annotations --exp-name 20260812_foldbox_rtc_h50_bs32_base_disabled_preflight --num-train-steps 101 --batch-size 32 --num-workers 8 --save-interval 1000 --keep-period 1000 --log-interval 100 --no-wandb-enabled --ema-decay None
```

Expected: Step 0 and Step 100 loss remain in the normal tens-scale range, with no multi-million spikes. `train.py` also saves the final step, so retain and inspect the valid step-100 checkpoint rather than assuming the interval suppresses it.

- [x] **Step 2: Run the full focused verification suite**

Run:

```bash
uv run pytest src/openpi/policies/spiritai_policy_test.py src/openpi/policies/spiritai_bridge_test.py src/openpi/training/config_test.py src/openpi/rtc/runtime_config_test.py -q
uv run ruff check src/openpi/policies/spiritai_policy.py src/openpi/training/config.py src/openpi/policies/spiritai_policy_test.py src/openpi/policies/spiritai_bridge_test.py src/openpi/training/config_test.py
```

Expected: all tests and Ruff checks pass.

### Plan self-review

- [x] The transform runs before `Normalize`, and `compute_norm_stats.py` uses the same input transforms.
- [x] The target remains 25D Cartesian, preserving model padding, serving metadata, and bridge compatibility.
- [x] The deployment limit protects the actual 23D robot command even if a policy output contains base values.
- [x] No raw dataset file, LoRA parameter, RTC delay, batch size, horizon, or training schedule is changed.
