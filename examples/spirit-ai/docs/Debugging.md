# Debugging Notes

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
