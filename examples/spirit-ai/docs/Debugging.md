# Debugging Notes

## multiscale 训练中 torchcodec 读视频尾帧失败

**问题**：pi0.5 微调跑到中途后 DataLoader worker 报错：

```text
RuntimeError: Requested next frame while there are no more frames left to decode.
```

一次复现中，最后正常日志在 `Step 17200` 附近，实际坏样本是 multiscale dataset 的 `episode=1519`、`global_idx=694552`、`timestamp=6.833333333333333`。6 路 camera 都无法用 torchcodec 解码这个最后 timestamp。

**原因**：旧的 `build-multiscale --video_mode slice` 使用 `ffmpeg -c copy` 进行 stream-copy 切片。这个方式很快，但不是 frame-exact。它可能生成 header 显示 `nb_frames=206`、duration 看起来足够，但真实可解码帧只有 205 帧，最后一帧 timestamp 是 `6.800000`。parquet 仍保留 `6.833333` 这一行，训练时 LeRobot/torchcodec 按 timestamp 读取尾帧就会越界。

**排查**：

```bash
uv run python examples/spirit-ai/dataset_transform.py verify-video-sync \
    --dataset_dir /path/to/multiscale_dataset \
    --episode_index 1519 \
    --strict_frame_count
```

如果是同类问题，会看到 decoded frame count 小于 parquet rows，或者 `torchcodec cannot decode last parquet timestamp`。

**解决**：重新 build 新 dataset，不要修补当前坏 dataset。推荐使用 reencode 切片并开启 strict sync：

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

build 完后再跑完整检查：

```bash
uv run python examples/spirit-ai/dataset_transform.py verify-video-sync \
    --dataset_dir /path/to/new_multiscale_dataset \
    --strict_frame_count
```

`--video_slice_codec copy` 只建议用于快速实验，不建议用于训练数据。

## checkpoint 写入时 tmux/session 被 systemd-oomd 杀掉

**问题**：训练能正常跑到 `Step 5000`，但第一次保存 checkpoint 时任务突然退出，tmux session 也消失。wandb `output.log` 没有 Python traceback，只停在 Orbax checkpoint 写入附近：

```text
Saving checkpoint at step 5000
Started async saving checkpoint ...
Transferring arrays to host memory ...
Wrote 71 array_metadata.ArrayMetadata ...
```

对应时间点的 user journal 出现：

```text
systemd-oomd killed 468 process(es) in this unit.
```

另一次失败同样发生在 step 5000 checkpoint 写入阶段，并有：

```text
systemd-oomd killed 436 process(es) in this unit.
```

**原因**：这不是 dataset/video 读取问题，也不是磁盘空间不足。Orbax 保存 checkpoint 时会把 JAX 参数和 train state 搬到 host memory 并写盘，造成系统内存瞬时峰值。训练常驻内存、dataloader workers、视频解码预取、JAX/XLA buffers、wandb、桌面/terminal 进程和 checkpoint host-copy buffer 叠加后，触发了 `systemd-oomd`。如果训练是在 GUI terminal/VTE scope 里启动，`systemd-oomd` 会杀掉整个 `vte-spawn-...scope`，里面的 tmux server 也会一起死。

一次成功 checkpoint 的日志显示写入量大约是：

```text
params: 7.2 GiB
train_state: 3.5 GiB
```

实际保存过程的 RAM 峰值会高于写盘体积。没有 swap 时风险更高。

**排查**：

```bash
free -h
journalctl --user --since '2026-06-08 01:15:00' --until '2026-06-08 01:30:00' --no-pager | rg 'systemd-oomd|vte-spawn'
ls -lah checkpoints/<config_name>/<exp_name>/
find checkpoints/<config_name>/<exp_name> -maxdepth 2 -type d -name '*orbax-checkpoint-tmp*'
```

如果 checkpoint 目录只留下 `5000.orbax-checkpoint-tmp-*`，没有最终 `5000/`，说明保存还没 finalize 就被杀了。

**解决**：

1. 添加 swap。64GB 是最低建议，磁盘空间充足时可以用 128GB：

```bash
sudo fallocate -l 64G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
free -h
```

需要重启后保留 swap 时写入 `/etc/fstab`：

```bash
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

2. 降低 dataloader workers。默认是 `6`，建议先用 `2`，折中可用 `4`：

```bash
--num-workers 2
```

3. 重新训练时使用 `--overwrite`，不要从未完成的 `.orbax-checkpoint-tmp-*` 目录恢复。

4. 启动前确认 GPU driver 正常：

```bash
nvidia-smi
```

5. 如果加 swap 和降低 workers 后仍在 checkpoint 阶段 OOM，下一步应改 checkpoint 策略，只保存 LoRA/trainable params，而不是完整 base model params/train state。

## wandb login 失败：API key must be 40 characters long

**问题**：`wandb login` 粘贴 API key 后报错 `ValueError: API key must be 40 characters long, yours was 86`。

**原因**：wandb CLI 交互式写入 key 时校验逻辑与新版 API key 格式不兼容。

**解决**：绕过交互式 CLI，手动写入 `~/.netrc`：

```bash
cat >> ~/.netrc << 'EOF'
machine api.wandb.ai
  login user
  password <你的API_KEY>
EOF
```

**验证**：

```bash
python -c "import wandb; print('logged_in' if wandb.api.api_key else 'not logged in')"
```

或者跑一个测试 run：

```bash
python -c "import wandb; wandb.init(project='test', mode='online').finish(); print('OK')"
```

## 训练时 terminal 崩溃导致 checkpoint 无法保存

**问题**：训练过程中 checkpoint 写入时间较长，terminal 断连或崩溃后训练进程被杀掉，可能导致权重文件没有写入完成或保存失败。

**原因**：训练进程直接挂在交互式 terminal 下，terminal 会话异常退出时会连带终止训练进程；如果刚好发生在 checkpoint 写盘阶段，就可能留下不完整的 checkpoint。

**解决**：用 `tmux` 启动训练任务，让训练进程脱离当前 terminal 会话：

```bash
tmux new -s spiritai_train

uv run python scripts/train.py pi05_spiritai_lora \
    --exp_name my_experiment \
    --overwrite
```

启动后可按 `Ctrl-b` 再按 `d` 退出 tmux 会话，训练会继续在后台运行。需要查看进度时重新进入：

```bash
tmux attach -t spiritai_train
```

**验证**：确认训练日志正常继续输出，并在 `checkpoints/pi05_spiritai_lora/<exp_name>/` 下看到完整的 checkpoint 目录。
