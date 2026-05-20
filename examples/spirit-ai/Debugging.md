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
