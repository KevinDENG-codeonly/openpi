# FoldBox PI0.5 LoRA RTC 部署 Benchmark 报告

## 结论摘要

本报告记录 FoldBox PI0.5 LoRA 权重在 RTX PRO 5000 Blackwell 48 GB 上的部署基准测试结果。测试对象为配置 `pi05_spiritai_cart_lora_h50_20260805_14annotations`，实验 `20260812_foldbox_rtc_h50_bs32_lora_90k` 的 `85000` checkpoint。该 checkpoint 的 Orbax 元数据、推理参数、训练状态与数据归一化资产均完整；模型能够以训练时 RTC（training-time RTC）方式运行，并正确广告 `training_time_v1` 能力。

结论需要区分“30 Hz 动作控制”和“30 Hz 重规划”。在本机稳态测试中，完整 WebSocket policy RPC 的 p99 为 175.35 ms，显著高于 33.3 ms，因此模型不能在每个控制 tick 都完成一次重规划。不过，现有运行时并不要求这样做：它以 30 Hz 逐行下发动作，同时每 15 个 tick 发起一次异步重规划。15 tick 对应 500 ms 的 RTC 延迟窗口，本次 RPC p99 仍有约 325 ms 余量。因此，该权重和当前 RTC 架构支持 **30 Hz 动作下发 + 约 2 Hz 异步重规划**，而不支持 30 Hz 同步逐帧推理。

首次推理的 JIT 编译约需 16 秒，是当前部署路径的首要风险。运行 YAML 中 `rtc.initial_inference_timeout_s` 的默认值为 10 秒，低于实测冷启动时延；若不预热或不提高该超时，真实 bridge 将在首次 action plan 生成前失败。除此之外，真实机器人侧的“读取 observation + command ACK”尚未实测，故不能仅凭本报告宣布端到端真实机器人控制已通过。

## 测试对象与边界

测试 checkpoint 位于 `checkpoints/pi05_spiritai_cart_lora_h50_20260805_14annotations/20260812_foldbox_rtc_h50_bs32_lora_90k/85000`。其训练配置为 Pi0.5 LoRA，`action_horizon=50`、模型 action dimension 为 32，并启用了 `rtc_training`，训练上限为 `max_delay_steps=20`。部署侧采用 Cartesian 输出：模型原始输出为 `(50, 32)`，经过输出变换后为 `(50, 25)` 的 SpiritAI Cartesian action。

测试机器为 NVIDIA RTX PRO 5000 Blackwell 48 GB。初始 GPU 占用为桌面服务的 373 MiB；模型加载并完成推理后，GPU 总占用约为 9.0 GiB，意味着此部署进程约增加 8.4 GiB。测试进程退出后显存恢复到基线，未观察到遗留 GPU 计算进程。

本次验证使用仓库内置的 `make_spiritai_cartesian_example()` 创建 fake observation。它与目标 policy 的输入结构一致：包含 `cam_high`、`cam_left_wrist`、`cam_right_wrist` 三路 480×640 RGB 图像，以及 25D Cartesian 状态。该选择足以验证模型装载、输入变换、RTC 请求、前缀冻结、WebSocket 传输和时延；但图像内容不来自 FoldBox 训练数据集，不能代表模型对真实纸盒场景的识别质量、动作合理性或任务成功率。

## 测试方法

测试分为三层。第一层直接测量 JAX `sample_actions`，并在每次调用后显式同步 GPU，避免异步 dispatch 导致的虚低时延。第二层从 policy 的标准 `infer` 接口测量完整 in-process 推理，因而包含输入变换、采样、输出变换以及模型 action 到 SpiritAI Cartesian action 的映射。第三层启动临时 loopback WebSocket policy server，并从本机 client 发起 RTC 请求；这一层额外覆盖 MessagePack 序列化与反序列化、WebSocket 收发以及 server 调度。

所有稳态统计均在编译预热后采集 10 个样本。N=10 足以判断 33.3 ms 与 500 ms 两个量级相差悬殊的预算，但对尾部抖动的统计置信度有限。上线前仍应在生产网络、真实 observation 与持续负载下以至少 100 个样本复测。

## 稳态时延与显存结果

下表中的 `model forward` 是 GPU 同步后的采样时延；`完整 policy` 表示进程内的实际 policy 调用；`server policy` 是 server 内部调用 `policy.infer` 的时间；`WebSocket RPC` 是客户端看到的端到端时延。后者是部署判断中最应优先参考的 policy 时延。

| 测量路径（RTC，N=10） | 均值 | p50 | p95 | p99 | 最大值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPU 同步 model forward | 143.58 ms | 143.45 ms | 150.13 ms | 151.62 ms | 152.00 ms |
| 完整 in-process policy inference | 145.27 ms | 145.27 ms | 149.47 ms | 149.83 ms | 149.93 ms |
| server 内部 policy inference | 155.85 ms | 155.85 ms | 165.20 ms | 166.76 ms | 167.15 ms |
| loopback WebSocket RPC 端到端 | 164.52 ms | 163.49 ms | 173.73 ms | 175.35 ms | 175.76 ms |

完整 RPC 与 server 内部推理之间的差值约为数毫秒到十余毫秒，来自 loopback 上的封包、socket 收发和 client 解包。该差值不代表跨机器网络时延；若 policy server 与 robot bridge 分处不同机器，还必须追加生产网络的测量。

冷启动与稳态必须分开看待。checkpoint 参数恢复约为 4.78 秒；第一次编译并执行 model forward 约为 15.57 秒；首次 loopback RTC RPC 约为 16.39 秒。之后，单次稳态 RTC RPC 降至约 162–165 ms。checkpoint 加载阶段 GPU 功耗较低是预期行为，因为其主要工作是存储读取、CPU 端反序列化和权重传输，而 GPU 上的主要编译与计算发生在第一笔推理中。测试使用 `XLA_PYTHON_CLIENT_PREALLOCATE=false`，也避免了 JAX 在启动时预留大块显存。

## RTC 功能验证

为验证 RTC 不是仅有 metadata，而是实际进入了模型路径，测试向 policy server 发送了一个 `delay_steps=15` 的请求。请求中的 `action_prefix` 形状为 `(50, 32)`，前 15 行固定为 `0.25`，其余行为零。server 返回的 raw `model_actions` 前 15 行与输入前缀逐元素完全相等，`prefix_exact_match=true`，最大绝对误差为 `0.0`。

这与 Pi0.5 的实现一致：采样过程会在每个 flow step 和最终输出处冻结前缀，因此已承诺给机器人执行的动作不会被新一轮采样改写。这个验证证明了 policy server、RTC envelope、capability 校验和模型硬前缀条件的端到端连通性；它不等同于评估固定前缀后的未来动作是否适合真实 FoldBox 任务。

## 30 Hz RTC 判断

在 30 Hz 下，每个控制 tick 为 33.3 ms。当前 YAML 的 `rpc_budget_fraction=0.7` 将机器人侧的“读状态 + 已接受 command ACK”预算限制为 23.33 ms。该预算属于机器人 RPC 安全门槛，并不要求异步 policy inference 在 23.33 ms 内完成。

模型与 policy server 的 p99 分别为约 152 ms 和 175 ms，显然不能支持每 tick 同步重新采样。从性能角度看，这并不是当前设计的失败条件。动作 horizon 为 50，因而一个 action plan 在 30 Hz 下覆盖约 1.667 秒；运行时会在 action plan 的第 15 个已确认 tick 发起替换请求，同时继续使用已安装 plan 的后续动作。`planned_max_steps=15` 给新请求 500 ms 的完成窗口，且小于训练时上限 20。实测稳态 RPC 约在 5 个 tick 内返回，远早于第 15 tick 的 deadline。

因此，在 robot RPC 满足 23.33 ms 预算的前提下，控制线程可继续以 30 Hz 发送单行命令，policy worker 则以约 2 Hz 的频率异步生成新的 50-step plan。若未来测得生产网络或系统负载造成 inference 接近 500 ms，应优先降低重规划频率，例如在任务允许更长开环时间时提高 `s_min`；不应把完整模型推理强行塞进 33.3 ms tick 中。

## 部署风险与建议

部署前必须解决冷启动问题。现有运行配置中的 `rtc.initial_inference_timeout_s=10.0` 小于实测 16.39 秒首次 RPC，单靠 `control.startup_delay_s=10.0` 并不能完成预热，因为它本身不会触发模型采样。建议在连接真实控制循环前完成一次不下发机器人动作的 policy warm-up；若运行路径仍要由 bridge 承担首次推理，则应将初始推理超时提高到至少 30 秒，以保留编译抖动余量。

保持 `control.source_hz=30.0`、`action_horizon=50`、`rtc.delay.planned_max_steps=15` 和 `rtc.s_min=10` 是当前最稳妥的组合。它提供 500 ms 的推理窗口和约 2 Hz 的重规划频率。将 `planned_max_steps` 降低至 10 虽然仍在本机实测范围内，但会把 deadline 缩短为约 333 ms，生产网络与机器人负载下的安全余量更小，当前证据不足以支持这一改变。若任务对突发视觉变化不敏感且需要减轻推理负载，可以评估提高 `s_min` 至 20；代价是重规划间隔增至约 0.67 秒，应以真实硬件运动质量作为决策依据。

无底盘目标在当前配置下是安全的，但需理解其协议边界。`max_base_speed=0.0` 会将底盘速度限制为零；若 robot server 广告 23D Cartesian command，bridge 仍可能在消息中携带三个零值底盘字段。若部署要求从协议层面完全不发送底盘字段，robot server 应只接受至多 20D Cartesian command，或者后续在 command-dimension 选择逻辑中增加显式的无底盘选项。

最后，真实部署仍缺少两类证据：一是机器人端 observation 读取与 command ACK 的 p50/p95/p99，以确认它们稳定低于 23.33 ms；二是真实 FoldBox 观测下的任务质量和 action plan 切换质量。只有这两项与本报告的 policy latency 共同满足要求，才能将“30 Hz 动作下发 + 异步 RTC 重规划”视为真实机器人部署通过。
