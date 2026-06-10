# Spirit AI moz1 Fine-Tuning and Deployment

This example contains the Spirit AI moz1 integration for LoRA fine-tuning of pi0.5, local LeRobot dataset preparation, policy serving, Python-client validation, and real-robot deployment through the Thor `robot_server` bridge.

## Scope

Use this README as the standard path for new Spirit AI experiments:

1. Prepare a LeRobot dataset and verify task prompts.
2. Optionally build a multiscale global-plus-subtask dataset.
3. Link the dataset into the local LeRobot cache.
4. Compute or reuse normalization statistics.
5. Run a checkpoint-readiness stress test.
6. Start the long LoRA fine-tuning run.
7. Serve and validate the checkpoint.
8. Run the Precision policy server to Thor `robot_server` bridge.

Current tuning notes and runbooks live in:

| File | Purpose |
|------|---------|
| [`docs/dataset_transform_cli.md`](docs/dataset_transform_cli.md) | Dataset transform CLI reference |
| [`docs/Debugging.md`](docs/Debugging.md) | Dataset/video/checkpoint debugging notes |
| [`docs/real_robot_notes.md`](docs/real_robot_notes.md) | Current Thor deployment status and motion findings |
| [`docs/motion_benchmark_plan.md`](docs/motion_benchmark_plan.md) | Motion benchmark matrix |

## Robot and Data Layouts

moz1 uses three RGB camera streams:

```text
cam_high, cam_left_wrist, cam_right_wrist
```

The joint-action baseline uses 27 real state/action dimensions, padded to pi0.5's 32-dim model action space:

| Segment | State key | Dims |
|---------|-----------|------|
| Left arm joints | `leftarm_state_joint_pos` | 7 |
| Left arm psi | `leftarm_state_psi` | 1 |
| Left gripper | `leftarm_gripper_state_pos` | 1 |
| Right arm joints | `rightarm_state_joint_pos` | 7 |
| Right arm psi | `rightarm_state_psi` | 1 |
| Right gripper | `rightarm_gripper_state_pos` | 1 |
| Torso joints | `torso_state_joint_pos` | 6 |
| Base speed | `base_state_speed` | 3 |
| Total | | 27 |

Action keys use the same ordering with `_cmd_` instead of `_state_`.

Cartesian-action configs replace arm/torso joint vectors with 6D Cartesian pose vectors:

| Layout | Real dims | Model dims | Policy output | Bridge command |
|--------|-----------|------------|---------------|----------------|
| Joint | 27 | 32 | `(T, 27)` | 25D/22D/16D joint commands; psi removed |
| Cartesian | 25 | 32 | `(T, 25)` | 23D/20D/14D Cartesian commands; psi removed |

## Prerequisites

- openpi installed from the repo root using `uv`
- A Spirit AI moz1 dataset in LeRobot v2.1 format
- Existing pi0.5 base checkpoint access through the configured weight loader
- Sufficient host RAM, swap, and disk for large Orbax checkpoints
- A GPU suitable for pi0.5 LoRA fine-tuning

## 1. Prepare the Dataset

Spirit AI datasets must have task prompts that match the intended deployment language. Folder names or task IDs are not useful model instructions.

### 1.1 Check or Repair Task Prompts

Inspect prompt metadata:

```bash
uv run python examples/spirit-ai/dataset_transform.py check \
    --dataset_dir /path/to/dataset \
    --default_prompt "Fold the cardboard sheet along the creases to form a box"
```

Write a repaired copy:

```bash
uv run python examples/spirit-ai/dataset_transform.py repair-instruction \
    --dataset_dir /path/to/dataset \
    --output_dir /path/to/dataset_repaired \
    --default_prompt "Fold the cardboard sheet along the creases to form a box" \
    --apply
```

Use the repaired dataset for norm stats and training. The old `check_instruction_manually.py` entry point remains as a compatibility wrapper; new workflows should call `dataset_transform.py`.

### 1.2 Optional Multiscale Dataset

For long-horizon tasks with valid subtask annotations in `meta/episodes.jsonl`, build a dataset that contains:

- full global episodes with an overview prompt;
- subtask-sliced episodes with prompts like `<overview>. Current step: <subtask>.`

This lets deployment still use a global prompt such as `fold the box`, while training also sees local stage supervision.

Subtask slicing is disabled by default because not every dataset has subtask metadata. Enable it explicitly with `--slice-episodes`.

```bash
uv run python examples/spirit-ai/dataset_transform.py build-multiscale \
    --dataset_dir /home/deng/Documents/dataset/20260512_FoldPaperBox_Moz1WB_MixedTask5+7_Slice \
    --output_dir /home/deng/Documents/dataset/20260512_FoldPaperBox_Moz1WB_MixedTask5+7_Slice_Multiscale \
    --global_prompt "Assemble the cardboard box by erecting the flat sheet and folding the side flaps." \
    --slice-episodes \
    --global_repeat 1 \
    --subtask_repeat 1 \
    --video_mode slice \
    --video_slice_codec reencode \
    --video_workers 6 \
    --overwrite
```

Important options:

| Option | Meaning |
|--------|---------|
| `--global_repeat` | Number of copies of full global episodes |
| `--subtask_repeat` | Number of copies of subtask-sliced episodes |
| `--video_mode link-full` | Slice parquet rows but hardlink full source videos |
| `--video_mode slice` | Physically slice videos for each derived episode |
| `--video_slice_codec reencode` | Recommended training path; validates decodable frame counts |
| `--video_slice_codec copy` | Faster but can leave MP4 tail-frame metadata mismatches |
| `--video_workers` | Number of camera videos processed in parallel |

With `global_repeat=1` and `subtask_repeat=1`, full-episode frames and subtask-sliced frames are approximately `1:1` by episode length.

Validate after building:

```bash
uv run python examples/spirit-ai/dataset_transform.py check \
    --dataset_dir /home/deng/Documents/dataset/20260512_FoldPaperBox_Moz1WB_MixedTask5+7_Slice_Multiscale \
    --default_prompt "Assemble the cardboard box by erecting the flat sheet and folding the side flaps." \
    --allow-derived-prompts
```

For physically sliced videos, also verify frame sync:

```bash
uv run python examples/spirit-ai/dataset_transform.py verify-video-sync \
    --dataset_dir /home/deng/Documents/dataset/20260512_FoldPaperBox_Moz1WB_MixedTask5+7_Slice_Multiscale \
    --strict_frame_count
```

## 2. Link the Dataset and Compute Norm Stats

openpi resolves LeRobot `repo_id` values under `~/.cache/huggingface/lerobot/`. For a local dataset, create a symlink that matches the `repo_id` in the train config.

```bash
mkdir -p ~/.cache/huggingface/lerobot/spiritai
ln -sfn /path/to/dataset_repaired \
    ~/.cache/huggingface/lerobot/spiritai/your_dataset_name
```

Then set the training config to the same repo ID:

```python
data=LeRobotSpiritaiDataConfig(
    repo_id="spiritai/your_dataset_name",
    ...
)
```

`repo_id` also determines where norm stats are stored under `assets/<config_name>/`. A repo ID containing `/` creates nested directories, for example:

```text
assets/pi05_spiritai_lora/spiritai/your_dataset_name/norm_stats.json
```

Compute norm stats from the repo root:

```bash
uv run python scripts/compute_norm_stats.py --config-name <config_name>
```

Examples:

```bash
uv run python scripts/compute_norm_stats.py --config-name pi05_spiritai_lora
uv run python scripts/compute_norm_stats.py --config-name pi05_spiritai_cart_lora_h30
uv run python scripts/compute_norm_stats.py --config-name pi05_spiritai_cart_lora_h50_multiscale
```

Norm stats can be reused across configs only when the underlying dataset layout and state/action dimensions match. Changing `action_horizon` alone does not change per-dimension statistics.

## 3. Train pi0.5 LoRA

Known Spirit AI configs live in [`src/openpi/training/config.py`](../../src/openpi/training/config.py).

| Config | Layout | Horizon | Dataset intent |
|--------|--------|---------|----------------|
| `pi05_spiritai_lora` | Joint 27D | 10 | Joint baseline |
| `pi05_spiritai_cart_lora_h30` | Cartesian 25D | 30 | Cartesian fold-box experiments |
| `pi05_spiritai_cart_lora_h30_20260512_mixed` | Cartesian 25D | 30 | 20260512 mixed no-slice dataset |
| `pi05_spiritai_cart_lora_h50_multiscale` | Cartesian 25D | 50 | 20260512 multiscale global/subtask dataset |

All listed pi0.5 LoRA configs load the pi0.5 base checkpoint, train LoRA adapters on the VLM/action expert, freeze non-LoRA weights, and use `ema_decay=None`.

### 3.1 Checkpoint Readiness

Before each new long fine-tuning run, verify that checkpoints can be finalized under the current host state. Saving pi0.5 checkpoints briefly stresses host RAM, swap, page cache, and filesystem writeback even when GPU memory is healthy.

Check disk, memory, and swap:

```bash
df -h / /tmp /home
free -h
swapon --show
```

Practical requirements:

- keep `/` and `/tmp` comfortably free, preferably `>50G`;
- keep large swapfiles on `/home`, not on the root filesystem;
- make sure `/home` has enough room for checkpoints;
- keep `--ema-decay None` for LoRA unless testing EMA deliberately.

Remove failed Orbax temporary checkpoints before reusing an experiment directory:

```bash
find checkpoints -type d -name "*.orbax-checkpoint-tmp-*" -print
```

After confirming the paths are failed leftovers:

```bash
find checkpoints -type d -name "*.orbax-checkpoint-tmp-*" -prune -exec rm -rf {} +
```

Run a short checkpoint stress test before the full job:

```bash
uv run python scripts/train.py <config_name> \
    --exp-name <exp_name> \
    --overwrite \
    --num-train-steps 1100 \
    --batch-size <batch_size> \
    --num-workers <workers> \
    --save-interval 1000 \
    --keep-period 1000 \
    --log-interval 100 \
    --wandb-enabled \
    --ema-decay None
```

The checkpoint is valid only after Orbax finalizes it. Seeing the progress bar move past the save step is not enough. Confirm the log contains:

```text
Finished asynchronous save
CheckpointManager Save Finalize is done on all hosts
```

Also verify the directory:

```bash
du -sh checkpoints/<config_name>/<exp_name>/1000
```

Monitor memory pressure near save steps when diagnosing:

```bash
watch -n 1 'date; free -h; cat /proc/pressure/memory; cat /sys/fs/cgroup/user.slice/user-1000.slice/memory.pressure'
```

If checkpoint finalization fails without a Python traceback and system logs show `systemd-oomd` killed the training scope, reduce `--num-workers`, keep `/` and `/tmp` free, and prefer running outside GUI-integrated terminal scopes.

### 3.2 Start Training

Use `tmux` for long training runs:

```bash
tmux new -s spiritai_train
```

Short smoke test:

```bash
uv run python scripts/train.py pi05_spiritai_lora \
    --exp-name smoke_test \
    --overwrite \
    --num-train-steps 10 \
    --batch-size 1 \
    --save-interval 10 \
    --log-interval 1 \
    --no-wandb-enabled \
    --ema-decay None
```

Current multiscale h50 diagnostic:

```bash
uv run python scripts/train.py pi05_spiritai_cart_lora_h50_multiscale \
    --exp-name 20260512_FoldPaperBox_multiscale_h50_50000stp \
    --overwrite \
    --num-train-steps 5100 \
    --batch-size 16 \
    --num-workers 4 \
    --save-interval 1000 \
    --keep-period 1000 \
    --log-interval 200 \
    --wandb-enabled \
    --ema-decay None
```

Current multiscale h50 full run:

```bash
uv run python scripts/train.py pi05_spiritai_cart_lora_h50_multiscale \
    --exp-name 20260512_FoldPaperBox_multiscale_h50_50000stp \
    --overwrite \
    --num-train-steps 50000 \
    --batch-size 16 \
    --num-workers 4 \
    --save-interval 5000 \
    --keep-period 5000 \
    --log-interval 200 \
    --wandb-enabled \
    --ema-decay None
```

Resume the same experiment only when a finalized checkpoint exists:

```bash
uv run python scripts/train.py pi05_spiritai_cart_lora_h50_multiscale \
    --exp-name 20260512_FoldPaperBox_multiscale_h50_50000stp \
    --resume \
    --num-train-steps 50000 \
    --batch-size 16 \
    --num-workers 4 \
    --save-interval 5000 \
    --keep-period 5000 \
    --log-interval 200 \
    --wandb-enabled \
    --ema-decay None
```

Detach from tmux with `Ctrl-b`, then `d`; reattach with:

```bash
tmux attach -t spiritai_train
```

### 3.3 Training Parameters

| Parameter | Guidance |
|-----------|----------|
| `model.action_horizon` | Number of future action frames predicted per policy call; do not change at inference without training that horizon |
| `model.action_dim` | Padded pi0.5 action dimension; Spirit AI uses 32 |
| `data.repo_id` | Must match the LeRobot symlink and norm stats path |
| `data.extra_delta_transform` | Keep `False` for current absolute command datasets |
| `batch_size` | Larger batches need more VRAM/host memory; validate with a short run |
| `num_workers` | More workers can improve loading speed but increase host memory pressure |
| `save_interval` | Lower values give more checkpoints but stress the host more often |
| `ema_decay` | Keep `None` for LoRA unless running an explicit EMA ablation |

Horizon selection is a deployment tradeoff. Longer chunks provide more future context but increase open-loop time. At `source_hz=15`, a 30-frame chunk lasts about `1.93s`; for contact-rich tasks, a common pattern is to predict a longer chunk but execute only a prefix before replanning.

## 4. Serve a Checkpoint

The `SPIRITAI` env mode is registered in [`scripts/serve_policy.py`](../../scripts/serve_policy.py).

Serve the default configured checkpoint:

```bash
uv run python scripts/serve_policy.py --env SPIRITAI --default_prompt "fold the paper box"
```

Serve a specific checkpoint:

```bash
uv run python scripts/serve_policy.py policy:checkpoint \
    --policy.config <config_name> \
    --policy.dir checkpoints/<config_name>/<exp_name>/<step> \
    --default_prompt "fold the paper box"
```

Cartesian h50 multiscale example:

```bash
uv run python scripts/serve_policy.py policy:checkpoint \
    --policy.config pi05_spiritai_cart_lora_h50_multiscale \
    --policy.dir checkpoints/pi05_spiritai_cart_lora_h50_multiscale/20260512_FoldPaperBox_multiscale_h50_50000stp/<step> \
    --default_prompt "Assemble the cardboard box by erecting the flat sheet and folding the side flaps."
```

The runtime prompt should match the training instruction distribution. For multiscale datasets, deployment usually uses the global task prompt, not a subtask-only prompt.

To make `--env SPIRITAI` point to a new checkpoint, update the `DEFAULT_CHECKPOINT` entry in `scripts/serve_policy.py`:

```python
EnvMode.SPIRITAI: Checkpoint(
    config="<config_name>",
    dir="checkpoints/<config_name>/<exp_name>/<step>",
)
```

## 5. Validate with the Python Client

Install the client package if needed:

```bash
cd $OPENPI_ROOT/packages/openpi-client
pip install -e .
```

Joint-policy observation keys:

| Key | Shape |
|-----|-------|
| `cam_high`, `cam_left_wrist`, `cam_right_wrist` | `(H, W, 3)` uint8 |
| `leftarm_state_joint_pos`, `rightarm_state_joint_pos` | `(7,)` |
| `leftarm_state_psi`, `rightarm_state_psi` | `(1,)` |
| `leftarm_gripper_state_pos`, `rightarm_gripper_state_pos` | `(1,)` |
| `torso_state_joint_pos` | `(6,)` |
| `base_state_speed` | `(3,)` |
| `prompt` | string |

Cartesian-policy observations replace joint state vectors with:

| Key | Shape |
|-----|-------|
| `leftarm_state_cart_pos`, `rightarm_state_cart_pos` | `(6,)` |
| `torso_state_cart_pos` | `(6,)` |

Policy-only smoke tests:

```python
from openpi.policies.spiritai_policy import make_spiritai_example
from openpi_client import websocket_client_policy

client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)
result = client.infer(make_spiritai_example())
print(result["actions"].shape)
```

```python
from openpi.policies.spiritai_policy import make_spiritai_cartesian_example
from openpi_client import websocket_client_policy

client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)
result = client.infer(make_spiritai_cartesian_example())
print(result["actions"].shape)
```

Expected shapes:

| Config layout | Output |
|---------------|--------|
| Joint horizon 10 | `(10, 27)` |
| Cartesian h30 | `(30, 25)` |
| Cartesian h50 | `(50, 25)` |

Current Spirit AI configs use `extra_delta_transform=False`. The returned actions are absolute command targets in raw dataset command semantics. Do not add the current state to `result["actions"]` unless you intentionally trained a delta-action policy and mirrored that transform at inference.

## 6. Real Robot Deployment Through Thor

The default robot path is split across two machines:

```text
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
  moz1 robot + GMSL cameras
```

`examples/spirit-ai/main.py` bridges the local policy server and remote `robot_server`. It does not talk to RealSense, ROS2, or the robot SDK directly. The old SDK environment remains in [`env.py`](env.py) for legacy/manual reference.

### 6.1 Start `robot_server` on Thor

```bash
sudo docker load -i /home/dengkevin/Documents/code/thor_robot_image/thor-robot_image.tar
sudo docker images | grep thor-robot

cd /home/dengkevin/Documents/code/robot_server_code
sudo bash run_robot.sh
sudo docker logs -f robot_server
```

Find the Thor IP reachable from Precision:

```bash
ip addr
```

### 6.2 Validate Thor from Precision

```bash
ping THOR_IP
nc -vz THOR_IP 8766

cd /home/dengkevin/Documents/code/robot_server_code
python test_connect.py --url ws://THOR_IP:8766
```

Expected current metadata:

| Field | Value |
|-------|-------|
| `structure` | `wholebody` |
| `joint_dim` | `25` |
| `cart_dim` | `23` |
| `accepted_joint_dims` | `[16, 22, 25]` |
| `accepted_cart_dims` | `[14, 20, 23]` |
| required cameras | `cam_high`, `cam_left_wrist`, `cam_right_wrist` |

Extra cameras are ignored by the bridge.

### 6.3 Start the Policy Server on Precision

```bash
cd /home/dengkevin/Documents/code/openpi
uv run python scripts/serve_policy.py --env SPIRITAI --default_prompt "fold the paper box"
```

For a Cartesian checkpoint:

```bash
uv run python scripts/serve_policy.py policy:checkpoint \
    --policy.config pi05_spiritai_cart_lora_h50_multiscale \
    --policy.dir checkpoints/pi05_spiritai_cart_lora_h50_multiscale/<exp_name>/<step> \
    --default_prompt "Assemble the cardboard box by erecting the flat sheet and folding the side flaps."
```

### 6.4 Run the Bridge

Joint policy example:

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

Cartesian policy example. Execute only a prefix of long predicted chunks and replan frequently:

```bash
cd /home/dengkevin/Documents/code/openpi
uv run python examples/spirit-ai/main.py \
    --policy-host localhost \
    --policy-port 8000 \
    --robot-url ws://THOR_IP:8766 \
    --prompt "Assemble the cardboard box by erecting the flat sheet and folding the side flaps." \
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

Bridge parameters most often tuned during deployment:

| Parameter | Role |
|-----------|------|
| `--policy-action-layout` | `joint` for 27D policy outputs, `cartesian` for 25D outputs |
| `--execute-steps` | Prefix length executed from each policy chunk |
| `--source-hz` | Action chunk frequency sent to `robot_server` |
| `--blend-steps` | Blend initial frames toward current robot state |
| `--rollback-guard-steps` / `--rollback-scale` | Suppress chunk-start rollback |
| `--prefetch-next-chunk` | Run the next policy inference while current chunk executes |
| `--max-arm-velocity-rad-s` / `--max-cart-translation-m-s` | Main motion clamps |
| `--max-base-speed` | Clamp base command dimensions |

Start with short runs (`--max-steps 3` to `10`). If `limited_fraction` is already around `0.45-0.60`, the bridge is materially changing the policy output; do not keep lowering velocity blindly.

## 7. RTC (Real-Time Chunking) Deployment

RTC adds replacement-style inpainting guidance in model action space to improve temporal consistency between consecutive action chunks. It is **experimental and disabled by default**.

### Key properties

- **No retraining required** - RTC operates at inference time using `return_model_actions` and the soft-mask inpainting target.
- **JAX pi0/pi0.5 only** - The PyTorch inference path currently rejects `rtc` kwargs.
- **Replacement inpainting, not full VJP guidance** - The current implementation shifts the previous model-space chunk and applies a soft mask (Eq. 5 from arXiv 2506.07339). Full VJP/GDM guidance is not yet implemented.
- **Uses existing prefetch/chunk loop** - RTC does not add a background inference thread; it piggybacks on the standard `--prefetch-next-chunk` mechanism.
- **Keep safety limits enabled** - RTC does not bypass motion limits, blend steps, or rollback suppression.

### Recommended starter flags

For the h50 multiscale Cartesian policy:

```bash
uv run python examples/spirit-ai/main.py \
    --policy-host localhost \
    --policy-port 8000 \
    --robot-url ws://THOR_IP:8766 \
    --prompt "Assemble the cardboard box by erecting the flat sheet and folding the side flaps." \
    --policy-action-layout cartesian \
    --execute-steps 15 \
    --enable-rtc \
    --rtc-s-min 5 \
    --rtc-beta 0.8 \
    --rtc-model-action-horizon 50 \
    --rtc-model-action-dim 32 \
    --enable-external-following \
    --source-hz 15 \
    --blend-steps 4 \
    --prefetch-next-chunk \
    --prefetch-delay-fraction 0.85 \
    --max-cart-translation-m-s 0.08 \
    --max-cart-rotation-rad-s 0.35 \
    --max-gripper-velocity-s 0.8 \
    --max-base-speed 0.05 \
    --max-steps 20
```

### RTC parameters

| Flag | Default | Meaning |
|------|---------|---------|
| `--enable-rtc` | `False` | Enable replacement-inpainting RTC guidance |
| `--rtc-s-min` | `5` | Minimum free (unconstrained) steps at the tail of each chunk |
| `--rtc-beta` | `0.8` | Guidance strength multiplier |
| `--rtc-initial-delay-steps` | `1` | Number of initial inferences without guidance |
| `--rtc-model-action-horizon` | `50` | Model action horizon (must match training config) |
| `--rtc-model-action-dim` | `32` | Model action dimension (must match training config) |

### Local testing note

If ROS pytest plugins interfere with running the RTC unit tests locally, disable plugin autoloading:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest src/openpi/rtc/rtc_test.py -q
```

## 8. Architecture Reference

Important files:

| File | Purpose |
|------|---------|
| [`dataset_transform.py`](dataset_transform.py) | Dataset prompt repair and multiscale builder |
| [`src/openpi/training/config.py`](../../src/openpi/training/config.py) | Spirit AI data configs and TrainConfigs |
| [`src/openpi/policies/spiritai_policy.py`](../../src/openpi/policies/spiritai_policy.py) | Policy input/output transforms and example observations |
| [`src/openpi/policies/spiritai_bridge.py`](../../src/openpi/policies/spiritai_bridge.py) | Robot-server observation mapping and command conversion |
| [`scripts/serve_policy.py`](../../scripts/serve_policy.py) | Policy server and `SPIRITAI` env mode |
| [`main.py`](main.py) | Precision policy server to Thor robot bridge |

Joint data flow:

1. LeRobot builds action sequences from joint `*_cmd_*` columns.
2. `SpiritaiInputs` concatenates 27D absolute state and action vectors.
3. Model transforms resize images, tokenize the prompt, and pad state/actions to 32D.
4. `SpiritaiOutputs` slices model output back to 27D.
5. `spiritai_bridge.py` drops psi dimensions and sends 25D/22D/16D joint commands.

Cartesian data flow:

1. LeRobot builds action sequences from Cartesian command, psi, gripper, torso, and base columns.
2. `SpiritaiCartesianInputs` concatenates 25D state and action vectors.
3. Model transforms pad state/actions to 32D.
4. `SpiritaiCartesianOutputs` slices model output back to 25D.
5. `spiritai_bridge.py` drops psi dimensions and sends 23D/20D/14D Cartesian commands.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `Norm stats not found` | `repo_id` does not match symlink/assets path | Recreate symlink, update config, recompute or copy norm stats |
| `Requested next frame while there are no more frames left to decode` | Video/parquet frame-count mismatch | Rebuild with `--video_slice_codec reencode`; run `verify-video-sync --strict_frame_count` |
| Checkpoint remains as `*.orbax-checkpoint-tmp-*` | Async save did not finalize | Check logs for `Finished asynchronous save`; inspect system logs for `systemd-oomd` |
| `ConnectionRefusedError` from policy client | Policy server not running | Start `serve_policy.py` and use the correct host/port |
| Missing robot cameras | Thor `robot_server` metadata lacks required camera names | Check `test_connect.py` and `run_robot.sh --camera-names` |
| Unsupported joint/cart metadata | Server accepted dims do not include bridge target dims | Check `accepted_joint_dims` / `accepted_cart_dims` |
| `Expected Cartesian actions with shape (T, 25)` | Cartesian bridge is receiving joint-policy output | Serve a Cartesian config and pass `--policy-action-layout cartesian` |
| `Expected actions with shape (T, 27)` | Joint bridge is receiving Cartesian-policy output | Use `--policy-action-layout cartesian` or serve a joint checkpoint |
| `accepted: false` ack | Server is busy or rejected command chunk | Check `docker logs -f robot_server`; reduce load and retry |
| `ModuleNotFoundError: openpi_client` | Client package missing | Install `packages/openpi-client` or run through `uv run` from repo root |

When switching datasets, repeat the full preparation path: repair prompts, rebuild/validate videos if needed, update symlink and `repo_id`, compute/copy norm stats, run checkpoint readiness, train, then serve the new finalized checkpoint.
