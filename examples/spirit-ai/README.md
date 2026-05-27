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
| `accepted_joint_dims` | `[16, 22, 25]` |
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

CLI parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--policy-host` | `localhost` | Policy server host |
| `--policy-port` | `8000` | Policy server port |
| `--robot-url` | `ws://172.16.0.30:8766` | Thor `robot_server` WebSocket URL |
| `--prompt` | `fold the paper box` | Task instruction; keep aligned with policy server `--default_prompt` |
| `--max-steps` | `2000` | Number of policy chunks to send |
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

### 7f. Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `ConnectionRefusedError` from policy client | Policy server not running | Start `uv run scripts/serve_policy.py --env SPIRITAI --default_prompt "fold the paper box"` |
| `robot_server is missing required cameras` | Camera names do not include policy keys | Check `test_connect.py` output and `run_robot.sh --camera-names` |
| `Unsupported robot_server joint metadata` | Server does not accept 16D/22D/25D joint commands | Check `accepted_joint_dims` from `test_connect.py` |
| `accepted: false` ack | Server is busy or rejected the chunk | Check `docker logs -f robot_server`; reduce load and retry |
| `ModuleNotFoundError: No module named 'openpi_client'` | Environment is missing workspace package | Run through `uv run` from the openpi repo root |

### 7g. Current Real Robot Deployment Notes

This section summarizes the current Thor `robot_server` deployment work and the latest motion tuning observations.

#### Deployment status

- Thor `robot_server` is running in Docker and listening on `ws://172.16.0.30:8766`.
- Precision connects to the local policy server at `localhost:8000` and bridges actions to Thor through [`main.py`](main.py).
- The current Thor metadata is `structure=wholebody`, `joint_dim=25`, and `accepted_joint_dims=[16, 22, 25]`.
- External following must be enabled before real motion. Use `--enable-external-following`; without it, the server accepts commands but arm joint states do not meaningfully follow.
- The bridge sends 25D joint commands in this layout:

```text
[left_joints(7), left_gripper(1),
 right_joints(7), right_gripper(1),
 torso_joints(6), base_speed(3)]
```

#### Work completed so far

- Replaced the old direct SDK/RealSense real robot entry path with a two-machine bridge.
- Added protocol support to request external following mode through `robot_server`.
- Added chunk-start smoothing:
  - `--blend-steps`
  - `--rollback-guard-steps`
  - `--rollback-scale`
- Added Precision-side motion limiting before sending commands to Thor:
  - `--max-arm-velocity-rad-s`
  - `--max-torso-velocity-rad-s`
  - `--max-gripper-velocity-s`
  - `--max-base-speed`
  - `--max-joint-accel-rad-s2`
- Added per-step logs for:
  - actual state delta since the previous chunk
  - command first-frame delta by group
  - raw vs limited max velocity
  - `limited_fraction`

#### Important behavior found during testing

The robot now moves reliably with external following enabled, but the latest real robot tests show a tradeoff:

- Strong velocity limiting reduces visible shaking.
- Strong velocity limiting also makes the executed path differ from the policy output, which can reduce continuity and task progress.
- `--prefetch-next-chunk` reduces waiting between chunks, but it can use an observation captured before the robot reaches the end of the current chunk. If the next policy output is based on that stale observation, the next chunk can fight the actual robot state.
- `--no-prefetch-next-chunk` made the motion slightly less shaky in testing, but continuity became worse because every chunk waits for the previous chunk to finish before doing the next policy call.
- Lowering `source_hz` slows the same 10-frame action chunk down. This can reduce speed-related shaking, but it also increases task latency.
- Increasing `source_hz` may improve responsiveness and training-frequency match, but it can increase chunk turnover and make boundary artifacts more visible unless the command path is smooth enough.

#### Recent benchmark observations

Manual scores from recent tests:

| Setting | Jitter | Continuity | Task intent | Notes |
|---------|--------|------------|-------------|-------|
| `source_hz=15`, `arm=0.28`, `torso=0.15`, `accel=0.8`, prefetch `0.85` | noticeable shake | medium-low | good | `limited_fraction` often around `0.45-0.60` |
| Same, prefetch `0.1` | worse shake | poor | unstable | prefetch likely too early; stale observation effect |
| Same, `--no-prefetch-next-chunk` | slightly less shake | poor | acceptable | less stale observation, but chunk-to-chunk waiting hurts continuity |

Earlier coarse benchmark trend:

| Variable | Observed trend |
|----------|----------------|
| Lower velocity | less shaking, but too low can make motion slow and discontinuous |
| Higher velocity | more responsive, but shaking returns more easily |
| `accel=0.0` | least shaking in earlier notes, but continuity was worse |
| `accel=0.8-1.2` | better continuity, slightly more shaking |
| `source_hz=12` | more continuous but weaker task intent |
| `source_hz=15` | best balance seen so far |
| `source_hz=20` | more responsive but can shake more |

#### Current interpretation

The remaining issue is probably not one single parameter. The likely causes are:

1. **Policy output is still high-velocity relative to safe real robot execution.** Logs often show `raw_max_vel` much larger than `limited_max_vel`.
2. **The limiter is changing a large fraction of the output.** `limited_fraction` around `0.45-0.60` means the executed trajectory is materially different from the model output.
3. **Chunk-level control is fighting continuous motion.** Each 10-frame policy chunk is smoothed locally, but there is no global continuous trajectory optimizer across chunks.
4. **Prefetch is a tradeoff, not a clear win.** Early prefetch can be stale; no prefetch adds inference gaps.
5. **Thor-side interpolation is still PCHIP per chunk.** It upsamples each received chunk to 120Hz, but it does not enforce global velocity, acceleration, or jerk continuity across chunk boundaries.

#### Recommended pause point

Do not keep increasing `--max-steps` until the short-run behavior is understood. Use `--max-steps 5` or `10` for further experiments.

The current safest reference command for analysis is:

```bash
uv run python examples/spirit-ai/main.py \
  --policy-host localhost \
  --policy-port 8000 \
  --robot-url ws://172.16.0.30:8766 \
  --prompt "Assemble the cardboard box by erecting the flat sheet and folding the side flaps" \
  --enable-external-following \
  --startup-delay-s 10 \
  --source-hz 15 \
  --blend-steps 4 \
  --rollback-guard-steps 4 \
  --rollback-scale 0.2 \
  --max-arm-velocity-rad-s 0.28 \
  --max-torso-velocity-rad-s 0.15 \
  --max-gripper-velocity-s 0.8 \
  --max-base-speed 0.05 \
  --max-joint-accel-rad-s2 0.8 \
  --no-prefetch-next-chunk \
  --max-steps 10
```

Use [`motion_benchmark_plan.md`](motion_benchmark_plan.md) for the full benchmark matrix.

#### Likely next engineering step

Parameter tuning alone is reaching diminishing returns. The next code-level improvement should be one of:

- Add a cross-chunk trajectory buffer that blends from the currently executing tail into the next policy chunk instead of sending independent chunks.
- Move velocity/acceleration/jerk limiting to Thor after PCHIP resampling, so the final 120Hz command stream is globally limited.
- Add log metrics for policy inference latency, chunk idle time, and boundary discontinuity between previous chunk tail and new chunk head.
- Compare against a recorded human/demo joint trajectory replay through the same `robot_server` path. If replay is smooth but policy is not, the issue is mostly policy output/noise. If replay also shakes, the issue is lower in the robot command/interpolation path.

## Architecture Reference

### Files

| File | Purpose |
|------|---------|
| [`scripts/serve_policy.py`](../../scripts/serve_policy.py) | Policy server with `SPIRITAI` env mode and `DEFAULT_CHECKPOINT` |
| [`src/openpi/policies/spiritai_policy.py`](../../src/openpi/policies/spiritai_policy.py) | Input/output transforms (`SpiritaiInputs`, `SpiritaiOutputs`) |
| [`src/openpi/policies/spiritai_bridge.py`](../../src/openpi/policies/spiritai_bridge.py) | `robot_server` observation mapping, metadata handling, msgpack codec, and 27D→joint command conversion |
| [`src/openpi/training/config.py`](../../src/openpi/training/config.py) | `LeRobotSpiritaiDataConfig` and `pi05_spiritai_lora` TrainConfig |
| [`examples/spirit-ai/check_instruction_manually.py`](check_instruction_manually.py) | Dataset instruction validation & repair utility |
| [`examples/spirit-ai/main.py`](main.py) | Default real robot bridge entry point for Precision policy server ↔ Thor `robot_server` |
| [`examples/spirit-ai/env.py`](env.py) | Legacy direct MOZ1 SDK environment kept for reference |

### Data Flow

1. **LeRobot** loads the dataset and creates action sequences from the 8 `*_cmd_*` columns (via `action_sequence_keys`)
2. **`SpiritaiInputs`** concatenates the 8 absolute state columns into a 27-dim `state` vector, parses 3 camera images, and concatenates raw `*_cmd_*` action columns into a 27-dim `actions` vector
3. **Model transforms** pad state/actions to 32 dims (π0.5's `action_dim`), resize images to 224×224, tokenize the prompt
4. **`SpiritaiOutputs`** slices the first 27 dims from the padded 32-dim model output
5. **`spiritai_bridge.py`** drops the two psi command dimensions and sends 25D, 22D, or 16D joint commands according to `robot_server` metadata

### Training Config

The `pi05_spiritai_lora` config fine-tunes π0.5 with LoRA adapters on both the vision-language backbone (`gemma_2b_lora`) and the action expert (`gemma_300m_lora`). Base weights are loaded from the official π0.5 checkpoint. All non-LoRA parameters are frozen.

### Switching Datasets

To train on a different Spirit AI dataset:

1. Prepare the new dataset (Step 1 above)
2. Update the symlink (Step 2 above) or create a new one
3. Update `repo_id` in the `pi05_spiritai_lora` config in `config.py`
4. Re-run `compute_norm_stats.py` (Step 3 above)
5. Start training in `tmux` (Step 4 above)
6. Update `DEFAULT_CHECKPOINT[EnvMode.SPIRITAI].dir` in `serve_policy.py` and serve the new model (Step 5 above)

For the full training-to-inference workflow, follow Steps 1-7 in order: prepare the dataset, create the symlink, update `repo_id`, compute norm stats, train, serve the checkpoint, then query it from the Python client or the real robot entry point.
