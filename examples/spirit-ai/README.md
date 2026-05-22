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
    --default_prompt "Your Prompt" \
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

This writes `norm_stats.json` to `./assets/pi05_spiritai_lora/<repo_id>/`. The output path is determined by `config.assets_dirs / data_config.repo_id`, where `assets_dirs` resolves to `./assets/<config_name>` (i.e. `./assets/pi05_spiritai_lora`) and `repo_id` comes from your `LeRobotSpiritaiDataConfig`. Make sure to run this command from the repo root so that `./assets` resolves correctly.

> **Tip:** You should see a log line like `Writing stats to: /path/to/openpi/assets/pi05_spiritai_lora/spiritai/your_dataset_name` — verify this matches the expected path before proceeding.

## Step 4: Run LoRA Fine-Tuning

```bash
uv run python scripts/train.py pi05_spiritai_lora \
    --exp_name my_experiment \
    --overwrite
```

The training script automatically loads `norm_stats.json` from the same path that `compute_norm_stats.py` wrote to (`config.assets_dirs / asset_id`, where `asset_id` defaults to `repo_id` when no custom `assets.asset_id` is set). As long as you run from the repo root, the norm stats from Step 3 will be found. You should see a log line like `Loaded norm stats from ...` confirming the file was found; if missing, you'll see `Norm stats not found in ..., skipping.` instead.

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

For example:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config pi05_spiritai_lora \
    --policy.dir checkpoints/pi05_spiritai_lora/20260424_FoldPaperBox_Moz1WB_NoSlice_repaired—_91ep_12000stp/11999 \
    --default_prompt "fold the paper box"
```

### 5c. Switching to a new trained model

When you train a new experiment, you have two options:

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

### 5d. Switching models checklist

1. Update `DEFAULT_CHECKPOINT[EnvMode.SPIRITAI].dir` in [`scripts/serve_policy.py`](../../scripts/serve_policy.py) (if using `--env SPIRITAI`)
2. Pass the correct `--default_prompt` for the new task
3. Restart the server
4. Verify the observation dict keys match what [`SpiritaiInputs`](../../src/openpi/policies/spiritai_policy.py) expects (see Step 6 below)

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

## Step 7: Real Robot Inference

This step runs the full closed-loop inference: Policy Server on GPU ↔ Moz1 robot via SDK + network.

### Architecture

```
GPU Server (serve_policy.py)
   ↕ WebSocket :8000
Robot Control Machine (main.py)
   ├── Moz SDK → capture_observation() → send_action()
   ↕ ROS2 / CycloneDDS (network cable)
Moz1 Robot (172.16.0.30)
```

*If the GPU server and robot control machine are the same host*, use `--args.host localhost`.

### 7a. Prerequisites

1. **Install Moz Robot SDK** (`mozrobot` wheel)
2. **Install openpi-client** (Step 6) and additional dependencies:

```bash
pip install scipy tyro typing_extensions -i https://pypi.tuna.tsinghua.edu.cn/simple/ --trusted-host pypi.tuna.tsinghua.edu.cn
```

3. **Set PYTHONPATH** (required for both `openpi_client` and the spirit-ai example):

```bash
export PYTHONPATH=$(pwd):$(pwd)/packages/openpi-client/src:$PYTHONPATH
```

4. **ROS2 environment** (required by Moz SDK):

```bash
source /opt/ros/humble/setup.bash
```

5. **Network configuration** — see 7c below.

### 7b. Run

Terminal 1 — Start Policy Server (in `uv` managed environment):

```bash
uv run scripts/serve_policy.py --env SPIRITAI --default_prompt "fold the paper box"
```

Terminal 2 — Start Robot Client (in system Python with ROS2):

```bash
export PYTHONPATH=$(pwd):$(pwd)/packages/openpi-client/src:$PYTHONPATH
source /opt/ros/humble/setup.bash

python3 examples/spirit-ai/main.py \
    --args.host localhost \
    --args.prompt "fold the paper box"
```

All CLI parameters (use `--args.<field>` prefix):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--args.host` | `0.0.0.0` | Policy Server IP (`localhost` if same machine) |
| `--args.port` | `8000` | Policy Server port |
| `--args.realsense_serials` | `230322270398,313522302626,230422271253` | Camera serial numbers (comma-separated) |
| `--args.camera_resolutions` | `320*240,320*240,320*240` | Camera resolutions |
| `--args.structure` | `wholebody` | Robot structure (`wholebody` / `wholebody_without_base` / `dualarm`) |
| `--args.prompt` | `fold the paper box` | Task instruction (should match server's `--default_prompt`) |
| `--args.action_horizon` | `40` | Steps per inference cycle (10 model steps × 4x resample) |
| `--args.resample_ratio` | `4.0` | Resample from model Hz to 120Hz control |
| `--args.num_episodes` | `1` | Number of episodes |
| `--args.max_episode_steps` | `2000` | Max steps per episode |

### 7c. Network Configuration (CycloneDDS)

The Moz SDK uses CycloneDDS (ROS2 middleware) to communicate with the robot. A `cyclonedds.xml` configuration file is required. Example at `/path/to/mozrobot-0.1.3/cyclonedds.xml`.

**Critical**: The `NetworkInterfaceAddress` must match **your server's IP** on the robot subnet, NOT the robot's IP.

```xml
<General>
    <!-- This must be YOUR machine's IP on the 172.16.0.x subnet -->
    <NetworkInterfaceAddress>172.16.0.50</NetworkInterfaceAddress>
    <AllowMulticast>false</AllowMulticast>
</General>
<Discovery>
    <Peers>
        <Peer address="172.16.0.30"/>  <!-- Robot controller IP -->
        <Peer address="127.0.0.1"/>     <!-- Local loopback -->
    </Peers>
</Discovery>
```

To find your machine's IP on the robot subnet:

```bash
ip addr | grep "172.16"
```

> **Common error**: `does not match an available interface` — means `NetworkInterfaceAddress` is wrong. Update it to your actual IP.

### 7d. How It Works

The inference loop runs at 120Hz:

1. **`env.get_observation()`** — Reads robot state + cameras via Moz SDK, adds `prompt`
2. **`ActionChunkResampleBroker.infer()`** — Sends observation to Policy Server via WebSocket, receives `(10, 27)` action chunk
3. **Split + Resample** — The 27-dim flat action is split into 8 named keys, then resampled from ~30Hz to 120Hz via PchipInterpolator (4x ratio → 40 control steps)
4. **`env.apply_action()`** — Each resampled step is sent to the robot as `send_action()` at 120Hz
5. After 40 steps, triggers next inference → repeat

**Data flow**:

```
Policy Server returns: {"actions": (10, 27)}
  ↓ split_actions()
  {"leftarm_cmd_joint_pos": (10,7), "leftarm_cmd_psi": (10,1), ..., "base_cmd_speed": (10,3)}
  ↓ _resample_action (4x interpolation)
  List of 40 dicts, each with named keys matching SDK send_action format
  ↓ robot.send_action() @ 120Hz
  Robot executes actions
```

**Psi handling**: Currently `leftarm_state_psi` and `rightarm_state_psi` are set to `np.zeros(1)` in `env.py`. Update if your model requires actual psi values.

### 7e. Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `does not match an available interface` | `cyclonedds.xml` NetworkInterfaceAddress wrong | Update to your server's IP on 172.16.0.x subnet |
| `No device connected` (camera) | RealSense cameras not plugged in or wrong serials | Check `rs-enumerate-devices`, verify serial numbers |
| `rcl node's rmw handle is invalid` | ROS2 DDS can't reach robot | Check network, verify `cyclonedds.xml`, ensure `source /opt/ros/humble/setup.bash` |
| `ModuleNotFoundError: No module named 'openpi_client'` | PYTHONPATH not set | `export PYTHONPATH=$(pwd):$(pwd)/packages/openpi-client/src:$PYTHONPATH` |
| `Unrecognized options: --host` | tyro requires nested prefix | Use `--args.host` instead of `--host` |
| `ConnectionRefusedError` | Policy Server not running | Start server first: `uv run scripts/serve_policy.py --env SPIRITAI` |

## Architecture Reference

### Files

| File | Purpose |
|------|---------|
| [`scripts/serve_policy.py`](../../scripts/serve_policy.py) | Policy server with `SPIRITAI` env mode and `DEFAULT_CHECKPOINT` |
| [`src/openpi/policies/spiritai_policy.py`](../../src/openpi/policies/spiritai_policy.py) | Input/output transforms (`SpiritaiInputs`, `SpiritaiOutputs`) |
| [`src/openpi/training/config.py`](../../src/openpi/training/config.py) | `LeRobotSpiritaiDataConfig` and `pi05_spiritai_lora` TrainConfig |
| [`examples/spirit-ai/check_instruction_manually.py`](check_instruction_manually.py) | Dataset instruction validation & repair utility |
| [`examples/spirit-ai/main.py`](main.py) | Real robot inference entry point |
| [`examples/spirit-ai/env.py`](env.py) | MOZ1 robot environment (SDK → observation/action) |
| [`packages/openpi-client/src/openpi_client/action_chunk_resample_broker.py`](../../packages/openpi-client/src/openpi_client/action_chunk_resample_broker.py) | Action chunk splitting (27→8 keys) + 30Hz→120Hz resampling |

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
6. Update `DEFAULT_CHECKPOINT[EnvMode.SPIRITAI].dir` in `serve_policy.py` and serve the new model (Step 5 above)

### Full Workflow: Training a New Model → Inference

1. Prepare & fix dataset instructions → Step 1
2. Create symlink → Step 2
3. Update `repo_id` in `config.py` → Step 2
4. Compute norm stats → Step 3
5. Train → Step 4
6. Update `DEFAULT_CHECKPOINT[EnvMode.SPIRITAI].dir` in `serve_policy.py` → Step 5c
7. Serve → `uv run scripts/serve_policy.py --env SPIRITAI --default_prompt "your task"`
8. Query from robot → Step 6
9. Real robot inference → Step 7
