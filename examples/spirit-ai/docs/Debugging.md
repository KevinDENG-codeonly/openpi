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
