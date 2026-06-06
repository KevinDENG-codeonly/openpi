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

This repo also includes a Cartesian-action fine-tuning config. In that variant, the left/right arm joint vectors are replaced with 6D Cartesian pose vectors:

| Variant | Arm command keys | Real action dims | Padded model dims |
|---------|------------------|------------------|-------------------|
| Joint baseline | `leftarm_cmd_joint_pos`, `rightarm_cmd_joint_pos` | 27 | 32 |
| Cartesian h30 | `leftarm_cmd_cart_pos`, `rightarm_cmd_cart_pos` | 25 | 32 |

Cameras: `cam_high` (overhead), `cam_left_wrist`, `cam_right_wrist`.

## Prerequisites

- openpi installed following the main [README](../../README.md) (uses `uv`)
- A dataset collected on the moz1 robot in **LeRobot v2.1** format
- GPU with ≥24 GB VRAM (e.g. NVIDIA 4090 / A5000)

## Current Workflow

Use this page as the main path from dataset preparation to real robot inference:

1. Prepare and repair dataset instructions.
2. Create the local LeRobot symlink and compute norm stats.
3. Fine-tune `pi05_spiritai_lora`.
4. Serve a checkpoint with `serve_policy.py`.
5. Validate policy output with the Python client.
6. Run real robot inference through the Thor `robot_server` bridge.

Operational notes and tuning history live in separate files:

| File | Purpose |
|------|---------|
| [`docs/real_robot_notes.md`](docs/real_robot_notes.md) | Current Thor deployment status, motion findings, and next engineering work |
| [`docs/motion_benchmark_plan.md`](docs/motion_benchmark_plan.md) | Parameter benchmark matrix for jitter, continuity, and task intent |

## Step 1: Prepare Dataset Instructions

Spirit AI datasets often use the folder name as the task text, which is not a useful language instruction for the model. Use `dataset_transform.py` to check and repair the dataset task text:

```bash
# Dry-run: inspect the dataset
uv run python examples/spirit-ai/dataset_transform.py check \
    --dataset_dir /path/to/your_dataset \
    --default_prompt "Your Prompt"

# Apply fix: write a repaired copy with a proper instruction
uv run python examples/spirit-ai/dataset_transform.py repair-instruction \
    --dataset_dir /path/to/your_dataset \
    --default_prompt "Fold the cardboard sheet along the creases to form a box" \
    --output_dir /path/to/your_dataset_repaired \
    --apply
```

Use the **repaired** dataset for all subsequent steps.

The old `check_instruction_manually.py` script is kept as a compatibility wrapper, but new workflows should call `dataset_transform.py` directly.

### 1a. Optional: Build a Multiscale Dataset

For long-horizon tasks with subtask annotation metadata, you can build a mixed dataset containing both:

- full global episodes with the overview instruction;
- subtask-sliced episodes with prompts of the form `<overview>. Current step: <subtask>.`

This keeps deployment aligned with a global prompt such as `fold the box`, while giving fine-tuning extra local supervision for each stage.

By default, `build-multiscale` does **not** use subtask annotations. This is intentional because not every Spirit AI dataset has subtask metadata. Add `--slice-episodes` only when the source dataset has valid annotation segments in `meta/episodes.jsonl`.

```bash
uv run python examples/spirit-ai/dataset_transform.py build-multiscale \
    --dataset_dir /home/deng/Documents/dataset/20260512_FoldPaperBox_Moz1WB_MixedTask5+7_Slice \
    --output_dir /home/deng/Documents/dataset/20260512_FoldPaperBox_Moz1WB_MixedTask5+7_Slice_Multiscale \
    --global_prompt "Assemble the cardboard box by erecting the flat sheet and folding the side flaps." \
    --slice-episodes \
    --global_repeat 1 \
    --subtask_repeat 1 \
    --video_mode slice \
    --video_workers 6 \
    --overwrite
```

With `global_repeat=1` and `subtask_repeat=1`, the total full-episode frames and total subtask-sliced frames are approximately `1:1`. If you omit `--video_mode slice`, the CLI uses `link-full`: parquet episodes are sliced, but videos are hardlinked to the original full recording and source timestamps are preserved. With `--video_mode slice`, each subtask episode gets physically sliced videos; `--video_workers` controls how many camera videos are sliced in parallel.

After building, validate the output:

```bash
uv run python examples/spirit-ai/dataset_transform.py check \
    --dataset_dir /home/deng/Documents/dataset/20260512_FoldPaperBox_Moz1WB_MixedTask5+7_Slice_Multiscale \
    --default_prompt "Assemble the cardboard box by erecting the flat sheet and folding the side flaps." \
    --allow-derived-prompts
```

For detailed CLI options, see [`docs/dataset_transform_cli.md`](docs/dataset_transform_cli.md).

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

For the Cartesian h30 config used by `20260424_FoldPaperBox_Moz1WB_Slice_repaired`, the local symlink should be:

```bash
mkdir -p ~/.cache/huggingface/lerobot/spiritai
ln -sfn /home/deng/Documents/dataset/20260424_FoldPaperBox_Moz1WB_Slice_repaired \
    ~/.cache/huggingface/lerobot/spiritai/20260424_FoldPaperBox_Moz1WB_Slice_repaired
```

The matching training config is `pi05_spiritai_cart_lora_h30`.

> **Note:** The `repo_id` becomes part of the norm_stats storage path under `assets/`. A `repo_id` with a `/` creates nested subdirectories (e.g. `assets/pi05_spiritai_lora/spiritai/your_dataset_name/norm_stats.json`). If you prefer a flatter path, use a `repo_id` without `/` (e.g. `"spiritai_your_dataset_name"`) and place the symlink directly at `~/.cache/huggingface/lerobot/spiritai_your_dataset_name`.

## Step 3: Compute Normalization Statistics

```bash
uv run python scripts/compute_norm_stats.py --config-name pi05_spiritai_lora
```

For the Cartesian h30 config, you can reuse Cartesian norm stats computed for the same dataset because the per-dimension state/action statistics do not depend on `action_horizon`. One practical option is to symlink the old h50 stats directory:

```bash
ln -sfn pi05_spiritai_cart_lora_h50 assets/pi05_spiritai_cart_lora_h30
```

If you need to recompute them from scratch for the h30 config, run:

```bash
uv run python scripts/compute_norm_stats.py --config-name pi05_spiritai_cart_lora_h30
```

This writes `norm_stats.json` to `./assets/pi05_spiritai_lora/<repo_id>/`. The output path is determined by `config.assets_dirs / data_config.repo_id`, where `assets_dirs` resolves to `./assets/<config_name>` (i.e. `./assets/pi05_spiritai_lora`) and `repo_id` comes from your `LeRobotSpiritaiDataConfig`. Make sure to run this command from the repo root so that `./assets` resolves correctly.

> **Tip:** You should see a log line like `Writing stats to: /path/to/openpi/assets/pi05_spiritai_lora/spiritai/your_dataset_name` — verify this matches the expected path before proceeding.

## Step 4: Run LoRA Fine-Tuning

Training should be started inside `tmux` so the process survives terminal disconnects or UI crashes. This is especially important when checkpoint writes take a long time: if the terminal session dies during a slow disk write, the weight file may be left incomplete or fail to save.

```bash
tmux new -s spiritai_train

uv run python scripts/train.py pi05_spiritai_lora \
    --exp_name my_experiment \
    --overwrite
```

Detach with `Ctrl-b` then `d`; reattach with `tmux attach -t spiritai_train`.

The training script automatically loads `norm_stats.json` from the same path that `compute_norm_stats.py` wrote to (`config.assets_dirs / asset_id`, where `asset_id` defaults to `repo_id` when no custom `assets.asset_id` is set). As long as you run from the repo root, the norm stats from Step 3 will be found. You should see a log line like `Loaded norm stats from ...` confirming the file was found; if missing, you'll see `Norm stats not found in ..., skipping.` instead.

Common overrides:

Run these commands inside the same `tmux` session:

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

Cartesian h30 smoke test:

```bash
uv run python scripts/train.py pi05_spiritai_cart_lora_h30 \
    --num_train_steps 10 \
    --batch_size 16 \
    --exp_name smoke_cart_h30 \
    --overwrite \
    --no-wandb_enabled \
    --save_interval 10 \
    --log_interval 1
```

Cartesian h30 full training:

```bash
uv run python scripts/train.py pi05_spiritai_cart_lora_h30 \
    --num_train_steps 12000 \
    --batch_size 16 \
    --exp_name 20260424_FoldPaperBox_cart_h30_12000stp \
    --overwrite \
    --save_interval 2000 \
    --log_interval 100
```

Checkpoints are saved to `checkpoints/pi05_spiritai_lora/<exp_name>/`.

### 4a. Fine-tuning parameters and action horizon

The current Spirit AI training config is [`pi05_spiritai_lora`](../../src/openpi/training/config.py). The most important fields are:

| Parameter | Current role |
|-----------|--------------|
| `model.action_horizon` | Number of future action frames predicted by one policy call |
| `model.action_dim` | Padded model action dimension; current SpiritAI uses 27 real dims padded to 32 |
| `data.repo_id` | LeRobot dataset used for training and norm stats |
| `data.extra_delta_transform` | Whether absolute actions are converted to deltas during training |
| `batch_size` | Number of training samples per optimizer step |
| `lr_schedule` | Learning-rate schedule for LoRA fine-tuning |
| `num_train_steps` | Total optimizer steps |
| `freeze_filter` | Which parameters stay frozen; current config trains LoRA adapters only |

For the current checkpoint, `action_horizon=10`, so the policy returns:

```text
(10, 27)
```

The Cartesian config `pi05_spiritai_cart_lora_h30` uses `action_horizon=30` and returns:

```text
(30, 25)
```

Those 25 values are Cartesian command-space values, not joint commands. For real-robot deployment, run the bridge with `--policy-action-layout cartesian` so it sends `kind="cart"` commands to `robot_server`.

The model architecture can support longer horizons. For example, π0 configs commonly use larger horizons, and `sample_actions()` generates tensors shaped by the configured `model.action_horizon`. However, a checkpoint trained with `action_horizon=10` should be treated as a 10-frame policy. Do not simply change the inference config to `30` and expect the existing checkpoint to produce reliable 30-frame behavior; the extra horizon would be outside the training distribution.

If longer action chunks are desired, train a new config or continue fine-tuning with a longer horizon:

```python
TrainConfig(
    name="pi05_spiritai_lora_h20",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=20,
        discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ),
    data=LeRobotSpiritaiDataConfig(...),
    ...
)
```

Recommended horizon experiment order:

| Horizon | Use case | Risk |
|---------|----------|------|
| `10` | Current baseline; frequent visual feedback | More chunk boundaries |
| `16` | First longer-horizon experiment | Moderate open-loop drift |
| `20` | Good candidate for reducing chunk-boundary artifacts | More open-loop execution |
| `30` | Only after 16/20 work well | Long open-loop duration; higher contact-task risk |
| `50` | Long-context Cartesian experiment; execute only a prefix | Very long open-loop if all frames are executed |

Longer horizon is not automatically better. At `source_hz=15`, a 10-frame chunk lasts about `9/15 = 0.6s`, while a 30-frame chunk lasts about `29/15 = 1.93s`. Fewer policy calls can improve continuity, but the robot also runs longer without fresh image feedback. For contact-rich tasks such as folding a paper box, a good deployment pattern is usually:

```text
predict 16-20 frames, execute only the first 6-10 frames, then replan
```

For `action_horizon=30`, the full chunk is about `29/15 = 1.93s` at `source_hz=15`. The recommended deployment pattern is to predict 30 frames but execute only the first 15-20 frames before replanning.

This receding-horizon setup keeps some future context while avoiding a long open-loop rollout.

Other fine-tuning parameters:

- `batch_size`: Larger values improve gradient stability but require more VRAM. With a 48GB inference/training GPU, increasing batch size may be possible, but it should be validated with a short smoke run.
- `num_train_steps`: More steps can improve task fit, but overtraining may reduce robustness. Track validation rollouts or at least policy-only output statistics across checkpoints.
- `lr_schedule.peak_lr`: Higher learning rate adapts faster but can destabilize LoRA fine-tuning. Current `2e-5` is conservative.
- `save_interval`: Controls checkpoint frequency. Shorter intervals make it easier to compare checkpoints during robot testing.
- `data.extra_delta_transform`: Must match action semantics. Current SpiritAI actions are absolute joint targets, so this stays `False`. If training a delta-action policy, robot-side inference must add the current state back with the same per-dimension mask.
- `prompt_from_task` / `--default_prompt`: Language must match the training instruction distribution. A mismatch can preserve low-level motion but weaken task intent.

## Step 5: Serve the Fine-Tuned Model

### 5a. Quick serve with the `SPIRITAI` env mode

The `SPIRITAI` env mode is registered in [`scripts/serve_policy.py`](../../scripts/serve_policy.py) with a default checkpoint. To serve it:

```bash
uv run scripts/serve_policy.py --env SPIRITAI --default_prompt "fold the paper box"
```

> **Important:** `--default_prompt` is not hardcoded. Always pass the prompt that matches your task at runtime.

### 5b. Specify a custom checkpoint directly

To serve a specific checkpoint without modifying `serve_policy.py`:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config pi05_spiritai_lora \
    --policy.dir checkpoints/pi05_spiritai_lora/<exp_name>/<step> \
    --default_prompt "fold the paper box"
```

For a Cartesian h30 checkpoint:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config pi05_spiritai_cart_lora_h30 \
    --policy.dir checkpoints/pi05_spiritai_cart_lora_h30/<exp_name>/<step> \
    --default_prompt "Assemble the cardboard box by erecting the flat sheet and folding the side flaps"
```

For example:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config pi05_spiritai_lora \
    --policy.dir checkpoints/pi05_spiritai_lora/20260424_FoldPaperBox_Moz1WB_NoSlice_repaired—_91ep_12000stp/11999 \
    --default_prompt "fold the paper box"
```

### 5c. Switching to a new trained model

When you train a new experiment, choose one of these options:

**Option A** — Update the `DEFAULT_CHECKPOINT` in `serve_policy.py` so `--env SPIRITAI` points to the new checkpoint:

```python
# In scripts/serve_policy.py, find the DEFAULT_CHECKPOINT dict and update:
EnvMode.SPIRITAI: Checkpoint(
    config="pi05_spiritai_lora",
    dir="checkpoints/pi05_spiritai_lora/<new_exp_name>/<new_step>",
),
```

Fields to update:

| Field | What to change |
|-------|---------------|
| `dir` | Path to the new checkpoint directory under `checkpoints/pi05_spiritai_lora/` |
| `config` | Only change if you created a new training config in `config.py`; otherwise keep `pi05_spiritai_lora` |

**Option B** — Skip modifying `serve_policy.py` and use `policy:checkpoint` with `--policy.dir` (see 5b above).

In both cases, pass the correct `--default_prompt`, restart the server, and verify the observation dict keys match what [`SpiritaiInputs`](../../src/openpi/policies/spiritai_policy.py) expects (see Step 6 below).

## Step 6: Inference with Python Client

Once the server is running (Step 5), you can query it from your robot code using the `openpi_client` package.

### Install the client

```bash
cd $OPENPI_ROOT/packages/openpi-client
pip install -e .
```

### Query the server

The observation dict must match the keys expected by [`SpiritaiInputs`](../../src/openpi/policies/spiritai_policy.py). The required keys are:

| Key | Type | Shape | Description |
|-----|------|-------|-------------|
| `cam_high` | `uint8` ndarray | `(H, W, 3)` | Overhead camera image |
| `cam_left_wrist` | `uint8` ndarray | `(H, W, 3)` | Left wrist camera image |
| `cam_right_wrist` | `uint8` ndarray | `(H, W, 3)` | Right wrist camera image |
| `leftarm_state_joint_pos` | `float32` ndarray | `(7,)` | Left arm joint positions |
| `leftarm_state_psi` | `float32` ndarray | `(1,)` | Left arm psi |
| `leftarm_gripper_state_pos` | `float32` ndarray | `(1,)` | Left gripper position |
| `rightarm_state_joint_pos` | `float32` ndarray | `(7,)` | Right arm joint positions |
| `rightarm_state_psi` | `float32` ndarray | `(1,)` | Right arm psi |
| `rightarm_gripper_state_pos` | `float32` ndarray | `(1,)` | Right gripper position |
| `torso_state_joint_pos` | `float32` ndarray | `(6,)` | Torso joint positions |
| `base_state_speed` | `float32` ndarray | `(3,)` | Base speed |
| `prompt` | `str` | — | Task instruction |

Example:

```python
from openpi_client import image_tools
from openpi_client import websocket_client_policy

client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)

observation = {
    "cam_high": image_tools.convert_to_uint8(image_tools.resize_with_pad(cam_high_img, 224, 224)),
    "cam_left_wrist": image_tools.convert_to_uint8(image_tools.resize_with_pad(cam_left_wrist_img, 224, 224)),
    "cam_right_wrist": image_tools.convert_to_uint8(image_tools.resize_with_pad(cam_right_wrist_img, 224, 224)),
    "leftarm_state_joint_pos": leftarm_joint_pos,        # (7,) float32
    "leftarm_state_psi": leftarm_psi,                    # (1,) float32
    "leftarm_gripper_state_pos": leftarm_gripper_pos,    # (1,) float32
    "rightarm_state_joint_pos": rightarm_joint_pos,      # (7,) float32
    "rightarm_state_psi": rightarm_psi,                  # (1,) float32
    "rightarm_gripper_state_pos": rightarm_gripper_pos,  # (1,) float32
    "torso_state_joint_pos": torso_joint_pos,            # (6,) float32
    "base_state_speed": base_speed,                      # (3,) float32
    "prompt": "fold the paper box",
}

# Returns {"actions": ndarray of shape (action_horizon, 27)}
result = client.infer(observation)
action_chunk = result["actions"]
```

The returned `action_chunk` has shape `(action_horizon, 27)` where `action_horizon=10`. The 27 action dimensions follow the same order as the state keys but use `_cmd_` instead of `_state_`:

| Dims | Description |
|------|-------------|
| 0–6 | Left arm joint commands |
| 7 | Left arm psi command |
| 8 | Left gripper command |
| 9–15 | Right arm joint commands |
| 16 | Right arm psi command |
| 17 | Right gripper command |
| 18–23 | Torso joint commands |
| 24–26 | Base speed commands |

For `pi05_spiritai_cart_lora_h30`, the Python client observation must use Cartesian state keys instead of joint state keys:

| Key | Type | Shape | Description |
|-----|------|-------|-------------|
| `leftarm_state_cart_pos` | `float32` ndarray | `(6,)` | Left arm Cartesian pose |
| `leftarm_state_psi` | `float32` ndarray | `(1,)` | Left arm psi |
| `leftarm_gripper_state_pos` | `float32` ndarray | `(1,)` | Left gripper position |
| `rightarm_state_cart_pos` | `float32` ndarray | `(6,)` | Right arm Cartesian pose |
| `rightarm_state_psi` | `float32` ndarray | `(1,)` | Right arm psi |
| `rightarm_gripper_state_pos` | `float32` ndarray | `(1,)` | Right gripper position |
| `torso_state_cart_pos` | `float32` ndarray | `(6,)` | Torso Cartesian pose |
| `base_state_speed` | `float32` ndarray | `(3,)` | Base speed |

The Cartesian policy returns `(30, 25)`:

| Dims | Description |
|------|-------------|
| 0–5 | Left arm Cartesian command |
| 6 | Left arm psi command |
| 7 | Left gripper command |
| 8–13 | Right arm Cartesian command |
| 14 | Right arm psi command |
| 15 | Right gripper command |
| 16–21 | Torso Cartesian command |
| 22–24 | Base speed command |

### Action semantics

For the current `pi05_spiritai_lora` config, `extra_delta_transform=False`. This means openpi does **not** convert the dataset actions into `cmd - state` deltas during training, and it does **not** add the current state back to the predicted actions during inference.

The model input `state` is the normalized form of the absolute robot state columns:

- arm joint positions and psi values are absolute positions;
- gripper state is an absolute discrete/open-close state (0/1 in the current Spirit AI data);
- torso joint positions are absolute positions;
- base state is the current base speed.

The policy output is the unnormalized 27-dim action chunk in the raw dataset `*_cmd_*` semantics. For the current Spirit AI datasets, the joint/psi/torso command columns are absolute command targets, and gripper commands are absolute 0/1 states. Do **not** add the current state to `result["actions"]` for this config; doing so would double-count the position commands.

If you intentionally train a delta-action model, enable the corresponding delta transform consistently during training and inference, or add the current state on the robot client side with a carefully chosen per-dimension mask.

> **Tip:** You typically call `client.infer()` every N steps and execute the predicted action chunk open-loop for the intermediate steps.

### Quick smoke test with random data

You can use the `make_spiritai_example()` helper from [`spiritai_policy.py`](../../src/openpi/policies/spiritai_policy.py) to generate a random observation for testing:

```python
from openpi.policies.spiritai_policy import make_spiritai_example
from openpi_client import websocket_client_policy

client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)
example = make_spiritai_example()
result = client.infer(example)
print("Actions shape:", result["actions"].shape)  # (10, 27)
```

For the Cartesian h30 config:

```python
from openpi.policies.spiritai_policy import make_spiritai_cartesian_example
from openpi_client import websocket_client_policy

client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)
example = make_spiritai_cartesian_example()
result = client.infer(example)
print("Actions shape:", result["actions"].shape)  # (30, 25)
```

## Step 7: Real Robot Inference via Thor robot_server

The default real robot path is now split across two machines:

```
Precision / GPU machine
  serve_policy.py             ws://localhost:8000
  examples/spirit-ai/main.py  bridge client
        |
        | WebSocket ws://THOR_IP:8766
        v
Thor / Jetson robot machine
  robot_server Docker container
        |
        | pymozrobot + GMSL + ROS2
        v
  MOZ1 robot + GMSL cameras
```

`examples/spirit-ai/main.py` no longer talks to RealSense, ROS2, or the robot SDK directly. It only bridges the local policy server and the remote `robot_server`. The old SDK environment remains in [`env.py`](env.py) for reference and legacy manual use.

### 7a. Start robot_server on Thor

On Thor, import the robot image and start the long-running container:

```bash
sudo docker load -i /home/dengkevin/Documents/code/thor_robot_image/thor-robot_image.tar
sudo docker images | grep thor-robot

cd /home/dengkevin/Documents/code/robot_server_code
sudo bash run_robot.sh
sudo docker logs -f robot_server
```

The logs should show the robot structure, GMSL camera readiness, and a WebSocket listener such as `ws://0.0.0.0:8766`.

Find the Thor IP that the Precision/GPU machine can reach:

```bash
ip addr
```

### 7b. Validate robot_server from Precision

From the Precision/GPU machine:

```bash
ping THOR_IP
nc -vz THOR_IP 8766

cd /home/dengkevin/Documents/code/robot_server_code
python test_connect.py --url ws://THOR_IP:8766
```

For the current Thor setup, the expected metadata is:

| Field | Value |
|-------|-------|
| `structure` | `wholebody` |
| `joint_dim` | `25` |
| `cart_dim` | `23` |
| `accepted_joint_dims` | `[16, 22, 25]` |
| `accepted_cart_dims` | `[14, 20, 23]` |
| required cameras | `cam_high`, `cam_left_wrist`, `cam_right_wrist` |

Extra cameras such as `cam_high_extra`, `cam_left_wrist_extra`, and `cam_right_wrist_extra` are ignored by the policy bridge.

### 7c. Start the policy server on Precision

```bash
cd /home/dengkevin/Documents/code/openpi
uv run scripts/serve_policy.py --env SPIRITAI --default_prompt "fold the paper box"
```

Optional policy-only smoke test:

```bash
uv run python - <<'PY'
from openpi.policies.spiritai_policy import make_spiritai_example
from openpi_client import websocket_client_policy

client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)
res = client.infer(make_spiritai_example())
print(res["actions"].shape)
PY
```

Expected output:

```text
(10, 27)
```

For a Cartesian h30 checkpoint:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config pi05_spiritai_cart_lora_h30 \
    --policy.dir checkpoints/pi05_spiritai_cart_lora_h30/<exp_name>/<step> \
    --default_prompt "Assemble the cardboard box by erecting the flat sheet and folding the side flaps"
```

Cartesian policy-only smoke test:

```bash
uv run python - <<'PY'
from openpi.policies.spiritai_policy import make_spiritai_cartesian_example
from openpi_client import websocket_client_policy

client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)
res = client.infer(make_spiritai_cartesian_example())
print(res["actions"].shape)
PY
```

Expected output:

```text
(30, 25)
```

### 7d. Run the bridge

Start with a short run:

```bash
cd /home/dengkevin/Documents/code/openpi
uv run python examples/spirit-ai/main.py \
    --policy-host localhost \
    --policy-port 8000 \
    --robot-url ws://THOR_IP:8766 \
    --prompt "fold the paper box" \
    --enable-external-following \
    --startup-delay-s 10 \
    --source-hz 15 \
    --blend-steps 4 \
    --rollback-guard-steps 4 \
    --rollback-scale 0.2 \
    --max-arm-velocity-rad-s 0.35 \
    --max-torso-velocity-rad-s 0.2 \
    --max-gripper-velocity-s 0.8 \
    --max-base-speed 0.05 \
    --prefetch-next-chunk \
    --prefetch-delay-fraction 0.85 \
    --max-steps 5
```

If `robot_server` accepts the chunks and the robot feedback looks correct, increase `--max-steps` gradually.

For a Cartesian h30 checkpoint, use the Cartesian bridge path. Start with a short, conservative run and execute only the first 15 frames from each 30-frame policy chunk:

```bash
cd /home/dengkevin/Documents/code/openpi
uv run python examples/spirit-ai/main.py \
    --policy-host localhost \
    --policy-port 8000 \
    --robot-url ws://THOR_IP:8766 \
    --prompt "Assemble the cardboard box by erecting the flat sheet and folding the side flaps" \
    --policy-action-layout cartesian \
    --execute-steps 15 \
    --enable-external-following \
    --startup-delay-s 10 \
    --source-hz 15 \
    --blend-steps 4 \
    --rollback-guard-steps 4 \
    --rollback-scale 0.2 \
    --max-cart-translation-m-s 0.08 \
    --max-cart-rotation-rad-s 0.35 \
    --max-torso-cart-translation-m-s 0.04 \
    --max-torso-cart-rotation-rad-s 0.2 \
    --max-gripper-velocity-s 0.8 \
    --max-base-speed 0.05 \
    --no-prefetch-next-chunk \
    --max-steps 3
```

In Cartesian mode, `main.py` maps robot observations with Cartesian state keys, converts the policy's `(30, 25)` action chunk to robot_server Cartesian commands, and sends:

```python
{"kind": "cart", "actions": cart_commands}
```

For a wholebody robot with `cart_dim=23`, `--execute-steps 15` sends `(15, 23)` to `robot_server`.

CLI parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--policy-host` | `localhost` | Policy server host |
| `--policy-port` | `8000` | Policy server port |
| `--robot-url` | `ws://172.16.0.30:8766` | Thor `robot_server` WebSocket URL |
| `--prompt` | `fold the paper box` | Task instruction; keep aligned with policy server `--default_prompt` |
| `--max-steps` | `2000` | Number of policy chunks to send |
| `--policy-action-layout` | `joint` | `joint` for 27D joint policy outputs, `cartesian` for 25D Cartesian policy outputs |
| `--execute-steps` | `None` | Optional prefix length to execute from each policy chunk; use `15` or `20` for Cartesian h30 |
| `--source-hz` | `15.0` | Source action chunk frequency sent to `robot_server` |
| `--busy-sleep-s` | `0.01` | Poll interval while waiting for `robot_server` to become idle |
| `--startup-delay-s` | `10.0` | Delay after both servers connect and before the first inference/action chunk |
| `--enable-external-following` | `False` | Ask `robot_server` to enable arm external following mode before inference |
| `--blend-steps` | `4` | Align the first command to current state and blend this many chunk-start frames |
| `--rollback-guard-steps` | `4` | Suppress arm/torso rollback over this many chunk-start frames |
| `--rollback-scale` | `0.2` | Fraction of detected rollback to keep during guarded chunk-start frames |
| `--prefetch-next-chunk` | `True` | Run next policy inference while the current chunk is executing |
| `--prefetch-delay-fraction` | `0.5` | Fraction of remaining chunk execution time to wait before prefetching the next observation |
| `--max-arm-velocity-rad-s` | `0.35` | Max adjacent-frame arm joint velocity before sending to `robot_server` |
| `--max-torso-velocity-rad-s` | `0.2` | Max adjacent-frame torso joint velocity before sending to `robot_server` |
| `--max-gripper-velocity-s` | `0.8` | Max adjacent-frame gripper command velocity before sending to `robot_server` |
| `--max-base-speed` | `0.05` | Absolute clamp for base speed command dims |
| `--max-joint-accel-rad-s2` | `0.0` | Optional adjacent-frame acceleration limit; `0.0` disables it |
| `--max-cart-translation-m-s` | `0.08` | Max adjacent-frame arm Cartesian translation velocity in Cartesian mode |
| `--max-cart-rotation-rad-s` | `0.35` | Max adjacent-frame arm Cartesian rotation-vector velocity in Cartesian mode |
| `--max-torso-cart-translation-m-s` | `0.04` | Max adjacent-frame torso Cartesian translation velocity in Cartesian mode |
| `--max-torso-cart-rotation-rad-s` | `0.2` | Max adjacent-frame torso Cartesian rotation-vector velocity in Cartesian mode |
| `--max-cart-accel` | `0.0` | Optional adjacent-frame Cartesian acceleration limit; `0.0` disables it |

### 7e. Bridge action layout

The policy returns `(10, 27)` in SpiritAI layout:

```text
[left_joints(7), left_psi(1), left_gripper(1),
 right_joints(7), right_psi(1), right_gripper(1),
 torso_joints(6), base_speed(3)]
```

`robot_server` joint commands do not include psi. The bridge chooses the widest supported joint command from metadata:

| Command dim | Layout |
|-------------|--------|
| `25` | `left_joints(7), left_gripper(1), right_joints(7), right_gripper(1), torso_joints(6), base_speed(3)` |
| `22` | same as 25D, without `base_speed(3)` |
| `16` | arms and grippers only |

For the current Thor metadata (`joint_dim=25`, `accepted_joint_dims=[16, 22, 25]`), the bridge sends 25D joint commands.

For Cartesian h30, the policy returns `(30, 25)` in SpiritAI Cartesian layout:

```text
[left_cart(6), left_psi(1), left_gripper(1),
 right_cart(6), right_psi(1), right_gripper(1),
 torso_cart(6), base_speed(3)]
```

`robot_server` Cartesian commands do not include psi. The bridge chooses the widest supported Cartesian command from metadata:

| Command dim | Layout |
|-------------|--------|
| `23` | `left_cart(6), left_gripper(1), right_cart(6), right_gripper(1), torso_cart(6), base_speed(3)` |
| `20` | same as 23D, without `base_speed(3)` |
| `14` | arms and grippers only |

For the current Thor metadata (`cart_dim=23`, `accepted_cart_dims=[14, 20, 23]`), the bridge sends 23D Cartesian commands. With `--execute-steps 15`, the command shape is `(15, 23)`.

### 7f. Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `ConnectionRefusedError` from policy client | Policy server not running | Start `uv run scripts/serve_policy.py --env SPIRITAI --default_prompt "fold the paper box"` |
| `robot_server is missing required cameras` | Camera names do not include policy keys | Check `test_connect.py` output and `run_robot.sh --camera-names` |
| `Unsupported robot_server joint metadata` | Server does not accept 16D/22D/25D joint commands | Check `accepted_joint_dims` from `test_connect.py` |
| `Unsupported robot_server Cartesian metadata` | Server does not accept 14D/20D/23D Cartesian commands | Check `accepted_cart_dims` from `test_connect.py` |
| `Expected Cartesian actions with shape (T, 25)` | Cartesian bridge is receiving a joint checkpoint output, or the wrong `--policy.config` is served | Serve `pi05_spiritai_cart_lora_h30` and use `--policy-action-layout cartesian` |
| `Expected actions with shape (T, 27)` | Joint bridge is receiving a Cartesian checkpoint output | Use `--policy-action-layout cartesian` or serve a joint checkpoint |
| `accepted: false` ack | Server is busy or rejected the chunk | Check `docker logs -f robot_server`; reduce load and retry |
| `ModuleNotFoundError: No module named 'openpi_client'` | Environment is missing workspace package | Run through `uv run` from the openpi repo root |

### 7g. Operational Notes and Benchmarking

Keep the main README focused on the standard deployment path. Use these companion notes for current real robot findings:

| File | Purpose |
|------|---------|
| [`docs/real_robot_notes.md`](docs/real_robot_notes.md) | Current Thor deployment status, motion findings, and recommended next engineering work |
| [`docs/motion_benchmark_plan.md`](docs/motion_benchmark_plan.md) | Parameter benchmark matrix for jitter, continuity, and task intent |

Current short summary:

- The robot moves reliably after `--enable-external-following`.
- Motion tuning is now limited more by chunk-to-chunk execution and control smoothness than by one obvious CLI parameter.
- Use short runs (`--max-steps 5` or `10`) while debugging.
- Do not keep lowering velocity blindly; if `limited_fraction` is already around `0.45-0.60`, the bridge is materially changing the policy output.

## Architecture Reference

### Files

| File | Purpose |
|------|---------|
| [`scripts/serve_policy.py`](../../scripts/serve_policy.py) | Policy server with `SPIRITAI` env mode and `DEFAULT_CHECKPOINT` |
| [`src/openpi/policies/spiritai_policy.py`](../../src/openpi/policies/spiritai_policy.py) | Input/output transforms (`SpiritaiInputs`, `SpiritaiOutputs`) |
| [`src/openpi/policies/spiritai_bridge.py`](../../src/openpi/policies/spiritai_bridge.py) | `robot_server` observation mapping, metadata handling, msgpack codec, 27D→joint conversion, and 25D→Cartesian conversion |
| [`src/openpi/training/config.py`](../../src/openpi/training/config.py) | `LeRobotSpiritaiDataConfig` and `pi05_spiritai_lora` TrainConfig |
| [`examples/spirit-ai/dataset_transform.py`](dataset_transform.py) | Dataset instruction validation, repair, and multiscale dataset builder |
| [`examples/spirit-ai/docs/dataset_transform_cli.md`](docs/dataset_transform_cli.md) | Dataset transform CLI reference |
| [`examples/spirit-ai/main.py`](main.py) | Default real robot bridge entry point for Precision policy server ↔ Thor `robot_server` |
| [`examples/spirit-ai/env.py`](env.py) | Legacy direct MOZ1 SDK environment kept for reference |
| [`examples/spirit-ai/docs/real_robot_notes.md`](docs/real_robot_notes.md) | Current deployment notes and motion-tuning interpretation |
| [`examples/spirit-ai/docs/motion_benchmark_plan.md`](docs/motion_benchmark_plan.md) | Motion smoothness benchmark matrix |

### Data Flow

1. **LeRobot** loads the dataset and creates action sequences from the 8 `*_cmd_*` columns (via `action_sequence_keys`)
2. **`SpiritaiInputs`** concatenates the 8 absolute state columns into a 27-dim `state` vector, parses 3 camera images, and concatenates raw `*_cmd_*` action columns into a 27-dim `actions` vector
3. **Model transforms** pad state/actions to 32 dims (π0.5's `action_dim`), resize images to 224×224, tokenize the prompt
4. **`SpiritaiOutputs`** slices the first 27 dims from the padded 32-dim model output
5. **`spiritai_bridge.py`** drops the two psi command dimensions and sends 25D, 22D, or 16D joint commands according to `robot_server` metadata

For the Cartesian h30 config:

1. **LeRobot** creates action sequences from `*_cmd_cart_pos`, psi, gripper, and base speed columns
2. **`SpiritaiCartesianInputs`** concatenates Cartesian state columns into a 25-dim `state` vector and Cartesian command columns into a 25-dim `actions` vector
3. **Model transforms** pad state/actions to 32 dims
4. **`SpiritaiCartesianOutputs`** slices the first 25 dims from the padded model output
5. **`spiritai_bridge.py`** drops the two psi command dimensions and sends 23D, 20D, or 14D Cartesian commands according to `robot_server` metadata

### Training Config

The `pi05_spiritai_lora` and `pi05_spiritai_cart_lora_h30` configs fine-tune π0.5 with LoRA adapters on both the vision-language backbone (`gemma_2b_lora`) and the action expert (`gemma_300m_lora`). Base weights are loaded from the official π0.5 checkpoint. All non-LoRA parameters are frozen.

For action horizon and fine-tuning parameter guidance, see [Step 4a](#4a-fine-tuning-parameters-and-action-horizon).

### Switching Datasets

To train on a different Spirit AI dataset:

1. Prepare the new dataset (Step 1 above)
2. Update the symlink (Step 2 above) or create a new one
3. Update `repo_id` in the `pi05_spiritai_lora` config in `config.py`
4. Re-run `compute_norm_stats.py` (Step 3 above)
5. Start training in `tmux` (Step 4 above)
6. Update `DEFAULT_CHECKPOINT[EnvMode.SPIRITAI].dir` in `serve_policy.py` and serve the new model (Step 5 above)

For the full training-to-inference workflow, follow Steps 1-7 in order: prepare the dataset, create the symlink, update `repo_id`, compute norm stats, train, serve the checkpoint, then query it from the Python client or the real robot entry point.
