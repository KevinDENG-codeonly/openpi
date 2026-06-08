# Debugging Notes

## TorchCodec Fails on the Last Video Frame in Multiscale Training

**Symptom:** pi0.5 fine-tuning fails in a DataLoader worker with:

```text
RuntimeError: Requested next frame while there are no more frames left to decode.
```

One reproduced case failed after the last healthy log near `Step 17200`. The bad sample was in the multiscale dataset at `episode=1519`, `global_idx=694552`, `timestamp=6.833333333333333`. All 6 camera videos failed to decode that final timestamp with TorchCodec.

**Cause:** the old `build-multiscale --video_mode slice` path used `ffmpeg -c copy` stream-copy slicing. This is fast but not frame-exact. It can produce an MP4 whose header reports enough frames, for example `nb_frames=206`, while the real decodable video has only 205 frames and ends at timestamp `6.800000`. The parquet episode still contains the final `6.833333` timestamp, so LeRobot/TorchCodec fails when it tries to read that tail frame.

**Diagnose:**

```bash
uv run python examples/spirit-ai/dataset_transform.py verify-video-sync \
    --dataset_dir /path/to/multiscale_dataset \
    --episode_index 1519 \
    --strict_frame_count
```

For the same issue, the decoded video frame count is smaller than the parquet row count, or the verifier reports that TorchCodec cannot decode the last parquet timestamp.

**Fix:** rebuild the dataset from the source dataset instead of patching the broken derived dataset. Use re-encoded slicing and strict sync validation:

```bash
uv run python examples/spirit-ai/dataset_transform.py build-multiscale \
    --dataset_dir /path/to/source_dataset \
    --output_dir /path/to/new_multiscale_dataset \
    --global_prompt "Assemble the cardboard box by erecting the flat sheet and folding the side flaps." \
    --slice-episodes \
    --global_repeat 1 \
    --subtask_repeat 1 \
    --video_mode slice \
    --video_slice_codec reencode \
    --video_workers 6 \
    --overwrite
```

Then validate the rebuilt dataset:

```bash
uv run python examples/spirit-ai/dataset_transform.py verify-video-sync \
    --dataset_dir /path/to/new_multiscale_dataset \
    --strict_frame_count
```

Use `--video_slice_codec copy` only for quick experiments, not for training datasets.

## Checkpoint Does Not Finalize: systemd-oomd, Root Swapfile, and Async Save

**Symptom:** training reaches a save step, but no final checkpoint directory appears. The task, terminal, or tmux scope may disappear. wandb `output.log` has no Python traceback and usually stops near Orbax checkpoint save/finalize logs:

```text
Saving checkpoint at step 1000
Started async saving checkpoint ...
Transferring arrays to host memory ...
Finished blocking save ... Continuing to save asynchronously ...
Starting CheckpointManager Save Finalize thread=save_finalize
```

The progress bar may continue to `1001/5100` or `1002/5100`, but that does not mean the checkpoint was saved. Orbax saves asynchronously: the main training loop can continue while the background save thread is still finalizing. A checkpoint is usable only after both lines appear:

```text
Finished asynchronous save
CheckpointManager Save Finalize is done on all hosts
```

Failed saves leave only a temporary directory:

```text
checkpoints/<config>/<exp>/1000.orbax-checkpoint-tmp-0
```

The matching system journal may show:

```text
systemd-oomd killed ... process(es) in this unit.
Killed ... due to memory pressure ... > 50.00% for > 20s with reclaim activity
```

**Cause:** this is not a dataset/video read error, and it is not "no memory in the checkpoint tmp directory." Orbax copies JAX params and train state to host memory and then writes them through page cache/writeback. A pi0.5 LoRA checkpoint has logged write volume around:

```text
params: 7.2 GiB
train_state: 3.5 GiB
```

The real host RAM, page cache, and writeback peak can be higher than the final checkpoint size. Persistent training memory, DataLoader workers, video decoding, JAX/XLA buffers, wandb, GUI terminal processes, and checkpoint host-copy buffers can combine into high memory pressure. If training was launched from a GUI terminal/VTE/Chromium scope, `systemd-oomd` can kill that whole scope, including the tmux server inside it.

In one reproduced case, `/swapfile` lived on the root filesystem and consumed 64GB, leaving only about 8GB free on `/` and `/tmp`. Checkpoints were written to `/home`, but root and `/tmp` pressure still worsened reclaim/writeback behavior. Moving swap to `/home/swapfile` freed `/` and `/tmp` from about 92% used to about 23% used; after that, the same `5100 steps / save_interval 1000` diagnostic run finalized the step 1000 checkpoint successfully.

**Diagnose:**

```bash
free -h
swapon --show
df -h / /tmp /home
journalctl --since 'YYYY-MM-DD HH:MM:SS' --until 'YYYY-MM-DD HH:MM:SS' --no-pager | rg 'systemd-oomd|oomd killed|memory pressure'
ls -lah checkpoints/<config_name>/<exp_name>/
find checkpoints/<config_name>/<exp_name> -maxdepth 2 -type d -name '*orbax-checkpoint-tmp*'
```

Watch PSI memory pressure around save steps:

```bash
watch -n 1 'date; free -h; cat /proc/pressure/memory; cat /sys/fs/cgroup/user.slice/user-1000.slice/memory.pressure'
```

If the checkpoint directory contains only `1000.orbax-checkpoint-tmp-*` or `5000.orbax-checkpoint-tmp-*`, and no final `1000/` or `5000/`, the save was interrupted before finalize.

**Fix:**

1. Keep `/`, `/tmp`, and `/home` healthy. Before long runs, leave at least tens of GB free on `/` and `/tmp`; `>50G` is preferable for large pi0.5 jobs.

2. If a large swapfile is on `/`, move it to `/home` while no training job is running:

```bash
sudo fallocate -l 64G /home/swapfile
sudo chmod 600 /home/swapfile
sudo mkswap /home/swapfile
sudo swapon /home/swapfile

sudo swapoff /swapfile
sudo cp /etc/fstab /etc/fstab.bak.$(date +%Y%m%d_%H%M%S)
sudo sed -i '/\/swapfile[[:space:]]/d;/\/home\/swapfile[[:space:]]/d' /etc/fstab
echo '/home/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
sudo rm /swapfile

swapon --show
df -h / /tmp /home
```

3. Remove failed Orbax temporary directories, and do not resume from them:

```bash
find checkpoints -type d -name "*.orbax-checkpoint-tmp-*" -print
find checkpoints -type d -name "*.orbax-checkpoint-tmp-*" -prune -exec rm -rf {} +
```

4. Reduce DataLoader workers. The default is `6`; use `2` for stability-first runs or `4` as a compromise:

```bash
--num-workers 2
```

5. Run a short checkpoint stress test before the long run:

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

6. If `systemd-oomd` still kills the job during checkpoint finalize, run training outside GUI-integrated terminal scopes, or temporarily disable `systemd-oomd` for one diagnostic run.

7. If the issue persists after the system-side fixes, consider changing the checkpoint strategy to save only LoRA/trainable params instead of full base-model params and train state.

## wandb Login Fails: API Key Must Be 40 Characters Long

**Symptom:** `wandb login` fails after pasting an API key:

```text
ValueError: API key must be 40 characters long, yours was 86
```

**Cause:** the interactive wandb CLI key validation is incompatible with the newer API key format in this environment.

**Fix:** bypass the interactive CLI and write `~/.netrc` manually:

```bash
cat >> ~/.netrc << 'EOF'
machine api.wandb.ai
  login user
  password <YOUR_API_KEY>
EOF
```

**Verify:**

```bash
python -c "import wandb; print('logged_in' if wandb.api.api_key else 'not logged in')"
```

Or start a small online run:

```bash
python -c "import wandb; wandb.init(project='test', mode='online').finish(); print('OK')"
```

## Terminal Crash Interrupts Checkpoint Save

**Symptom:** checkpoint writes take a long time, and if the terminal disconnects or crashes during the write, training is killed and the checkpoint is incomplete.

**Cause:** the training process is attached directly to an interactive terminal. If that terminal exits during checkpoint write/finalize, the process can be terminated before Orbax finalizes the checkpoint.

**Fix:** start long training jobs from `tmux`:

```bash
tmux new -s spiritai_train

uv run python scripts/train.py pi05_spiritai_lora \
    --exp-name my_experiment \
    --overwrite
```

Detach with `Ctrl-b`, then `d`. Reattach with:

```bash
tmux attach -t spiritai_train
```

**Verify:** logs should continue after detaching, and finalized checkpoint directories should appear under:

```text
checkpoints/<config_name>/<exp_name>/<step>/
```
