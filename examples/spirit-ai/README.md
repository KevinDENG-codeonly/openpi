# Spirit AI Humanoid Robot (moz1)

This example integrates Spirit AI's moz1 bimanual humanoid robot with openpi for LoRA fine-tuning of the π0.5 model.

## Robot Overview

The Spirit AI moz1 robot has:

| Body Part | State Keys | Dims |
|-----------|-----------|------|
| Left arm joints | `leftarm_state_joint_pos` | 7 |
| Left arm psi | `leftarm_state_psi` | 1 |
| Left gripper | `leftarm_gripper_state_pos` | 1 |
| Right arm joints | `rightarm_state_joint_pos` | 7 |
| Right arm psi | `rightarm_state_psi` | 1 |
| Right gripper | `rightarm_gripper_state_pos` | 1 |
| Torso | `torso_state_joint_pos` | 6 |
| Base | `base_state_speed` | 3 |
| **Total** | | **27** |

Action keys follow the same layout but use `_cmd_` instead of `_state_` (e.g. `leftarm_cmd_joint_pos`).

Cameras: `cam_high` (overhead), `cam_left_wrist`, `cam_right_wrist`.

## Prerequisites

- openpi installed following the main [README](../../README.md) (uses `uv`)
- A dataset collected on the moz1 robot in **LeRobot v2.1** format
- GPU with ≥24 GB VRAM (e.g. NVIDIA 4090 / A5000)

## Step 1: Check & Fix Dataset Instructions

Spirit AI datasets often use the folder name as the task text, which is not a useful language instruction for the model. Use the provided utility to check and fix this:

```bash
# Dry-run: inspect the dataset
uv run python examples/spirit-ai/check_instruction_manually.py \
    --dataset_dir /path/to/your_dataset

# Apply fix: write a repaired copy with a proper instruction
uv run python examples/spirit-ai/check_instruction_manually.py \
    --dataset_dir /path/to/your_dataset \
    --default_prompt "Fold the cardboard sheet along the creases to form a box" \
    --output_dir /path/to/your_dataset_repaired \
    --apply
```

Use the **repaired** dataset for all subsequent steps.

## Step 2: Create a Symlink for the Local Dataset

openpi uses LeRobot's dataset loading, which resolves `repo_id` against the LeRobot cache directory (`~/.cache/huggingface/lerobot/`). To use a local dataset, create a symlink:

```bash
mkdir -p ~/.cache/huggingface/lerobot/spiritai
ln -sfn /path/to/your_dataset_repaired \
    ~/.cache/huggingface/lerobot/spiritai/your_dataset_name
```

Then update the `repo_id` in the training config ([src/openpi/training/config.py](../../src/openpi/training/config.py)) to match:

```python
data=LeRobotSpiritaiDataConfig(
    repo_id="spiritai/your_dataset_name",
    ...
),
```

> **Note:** The `repo_id` becomes part of the norm_stats storage path under `assets/`. A `repo_id` with a `/` creates nested subdirectories (e.g. `assets/pi05_spiritai_lora/spiritai/your_dataset_name/norm_stats.json`). If you prefer a flatter path, use a `repo_id` without `/` (e.g. `"spiritai_your_dataset_name"`) and place the symlink directly at `~/.cache/huggingface/lerobot/spiritai_your_dataset_name`.

## Step 3: Compute Normalization Statistics

```bash
uv run python scripts/compute_norm_stats.py --config-name pi05_spiritai_lora
```

This writes `norm_stats.json` to `./assets/pi05_spiritai_lora/<repo_id>/`.

## Step 4: Run LoRA Fine-Tuning

```bash
uv run python scripts/train.py pi05_spiritai_lora \
    --exp_name my_experiment \
    --overwrite
```

Common overrides:

```bash
# Short smoke test (10 steps, batch size 1)
uv run python scripts/train.py pi05_spiritai_lora \
    --num_train_steps 10 \
    --batch_size 1 \
    --exp_name smoke_test \
    --overwrite \
    --no-wandb_enabled \
    --save_interval 10 \
    --log_interval 1

# Full training with custom steps
uv run python scripts/train.py pi05_spiritai_lora \
    --num_train_steps 1000 \
    --batch_size 10 \
    --exp_name 20260422_box_folding_27episodes_1000steps \
    --overwrite \
    --save_interval 200 \
    --log_interval 10
```

Checkpoints are saved to `checkpoints/pi05_spiritai_lora/<exp_name>/`.

## Step 5: Serve the Fine-Tuned Model

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config pi05_spiritai_lora \
    --policy.dir checkpoints/pi05_spiritai_lora/<exp_name>/<step>
```

## Architecture Reference

### Files

| File | Purpose |
|------|---------|
| [`src/openpi/policies/spiritai_policy.py`](../../src/openpi/policies/spiritai_policy.py) | Input/output transforms (`SpiritaiInputs`, `SpiritaiOutputs`) |
| [`src/openpi/training/config.py`](../../src/openpi/training/config.py) | `LeRobotSpiritaiDataConfig` and `pi05_spiritai_lora` TrainConfig |
| [`examples/spirit-ai/check_instruction_manually.py`](check_instruction_manually.py) | Dataset instruction validation & repair utility |

### Data Flow

1. **LeRobot** loads the dataset and creates action sequences from the 8 `*_cmd_*` columns (via `action_sequence_keys`)
2. **`SpiritaiInputs`** concatenates the 8 state columns into a 27-dim `state` vector, parses 3 camera images, and concatenates action columns into a 27-dim `actions` vector
3. **Model transforms** pad state/actions to 32 dims (π0.5's `action_dim`), resize images to 224×224, tokenize the prompt
4. **`SpiritaiOutputs`** slices the first 27 dims from the padded 32-dim model output

### Training Config

The `pi05_spiritai_lora` config fine-tunes π0.5 with LoRA adapters on both the vision-language backbone (`gemma_2b_lora`) and the action expert (`gemma_300m_lora`). Base weights are loaded from the official π0.5 checkpoint. All non-LoRA parameters are frozen.

### Switching Datasets

To train on a different Spirit AI dataset:

1. Prepare the new dataset (Step 1 above)
2. Update the symlink (Step 2 above) or create a new one
3. Update `repo_id` in the `pi05_spiritai_lora` config in `config.py`
4. Re-run `compute_norm_stats.py` (Step 3 above)
5. Start training (Step 4 above)
