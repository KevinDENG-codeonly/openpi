# Pi0.5 / SpiritAI training-time RTC 设计

**日期：** 2026-08-07
**状态：** 已实现
**范围：** JAX Pi0.5、SpiritAI Cartesian h=50。
**术语：** 论文和代码均使用 RTC（Real-Time Chunking）；本文不再使用 RCT。
**实施计划：** [training-time RTC implementation plan](../plans/2026-08-07-training-time-rtc.md)

## 实施状态与运行约束

本设计已按上述实施计划落地。SpiritAI 运行器只接受训练时启用
`rtc_training.enabled` 的 JAX Pi0.5 checkpoint；policy metadata 必须声明
`rtc_capabilities.algorithm: training_time_v1`，并提供模型 horizon、action dim 和
`training_max_delay_steps`。运行器从 metadata 获取模型维度，拒绝超过训练能力的
`planned_max_steps`。

在线采样唯一支持 hard action-prefix conditioning。VJP/PiGDM、`beta`、soft mask、
replacement inpainting 和其他 legacy RTC 路径均不可用。主线程只拥有 robot WebSocket，
单飞 worker 独占 policy 连接；每个控制 tick 至多发送一个经过安全限制的 robot action。

入口为 `uv run examples/spirit-ai/main.py`，默认读取相对 `main.py` 的 strict YAML profile；
`--config PATH` 覆盖该路径，`--dry-run` 抑制 command send。运行 RTC profile 时，robot 和
policy 必须使用非 TLS `ws://` endpoint：Linux total write-deadline 依赖 per-send
`MSG_DONTWAIT`，Python TLS socket 无法安全支持该机制，因此 `wss://` 会在任何 hardware 或
policy activity 前被拒绝。

在训练前必须在目标控制频率下测量端到端时延，并让训练
`max_delay_steps` 与部署 `rtc.delay.planned_max_steps` 采用相同的安全范围。每次运行报告
`dplan`、`dactual`、deadline misses、holds、plan-switch command delta、control frequency 和
end-to-end inference latency。YAML 同时配置 policy connection、initial inference、robot RPC
ACK 和 robot-idle timeout；超时、deadline 或 RPC-budget 失败会 fail closed，关闭 transport，
并在配置要求时发送一行 terminal hold 后停止。

## 1. 决策与目标

本次不再维护旧 checkpoint，也不实现 2506.07339 的 VJP/PiGDM、soft mask 或
`hybrid` 路径。目标是把 [training-time action conditioning](https://arxiv.org/html/2512.05964v2)
作为唯一的 RTC 算法路径，同时补齐它必需的异步在线执行器。

这意味着“推理时 RTC”仍然是必须改造的部分：推理器必须根据实际时延从旧 chunk
取出已经确定会执行的动作前缀，并在每一个 flow step 将其固定。区别仅在于，模型
已经在训练期学会这种条件分布，因此推理期没有梯度、VJP、`beta` 或软掩码开销。

本设计的成功标准：

- 对训练过 RTC 的 Pi0.5，异步执行时新 chunk 在切换点之前与旧 chunk 完全一致；
- 推理采样的前向次数与非 RTC Pi0.5 相同，且不含反向传播；
- 所有 RTC 参数从训练 config 或 SpiritAI YAML 读取，不再散落在启动命令；
- 时延超出训练/时间线约束时保持安全，不把不连续的 chunk 发给机器人；
- 保持 Pi0.5 已有 checkpoint 的**参数**可加载性，但不承诺其为 RTC checkpoint。

非目标：PyTorch Pi0、Pi0-FAST、VJP/PiGDM、soft mask、对历史 checkpoint 的运行时
兼容，以及在本次改造中改变机器人 server 协议。

## 2. 为什么选择 training-time RTC

2506.07339 的 inference-time RTC 通过 VJP 在每一步去噪时对完整 chunk 做软 inpainting；
它可支持未训练 checkpoint，却增加反向传播并使时延更高。2512.05964 将相同的硬动作
前缀条件直接放进训练：学习

```
p(A[t+d:H] | observation[t], A[t:t+d])
```

其中 `d` 是以控制步数衡量的推理时延。论文的真实机器人实验中，training-time 路径在
相同模型/5 个 flow steps 的设置下，端到端平均 108ms；inference-time VJP 路径为 135ms。
该数值不是本项目的性能承诺，但说明消除 VJP 是合理的主部署选择。

当前使用者会重新 fine-tune，且现有 checkpoint 不作为兼容目标。因此同时维护
`none / training / inference / hybrid` 四条路径只会增加调用、测试和故障排查复杂度，
而不提供对应收益。改造后仅保留：

- **普通模式**：训练 config 未启用 RTC 时，保持现有 Pi0.5 flow matching；
- **RTC 模式**：训练 config 启用后，部署必须使用 training-time RTC scheduler。

## 3. 训练设计

### 3.1 配置接口

在 `src/openpi/training/config.py` 增加不可变的 `RTCTrainingConfig`，并让
`TrainConfig` 持有 `rtc_training` 字段：

```python
@dataclasses.dataclass(frozen=True)
class RTCTrainingConfig:
    enabled: bool = False
    max_delay_steps: int = 0
    delay_distribution: Literal["uniform"] = "uniform"

    # 将来若确有数据支持，可增加 geometric；一期不加入多余分支。

@dataclasses.dataclass(frozen=True)
class TrainConfig:
    ...
    rtc_training: RTCTrainingConfig = dataclasses.field(default_factory=RTCTrainingConfig)
```

约束如下：

- `enabled=False` 是所有既有 config 的默认值；不开启时模型行为和损失定义不变；
- 一期只允许 `Pi0Config(pi05=True)`；其他模型启用时在启动训练前报清晰错误；
- `0 <= max_delay_steps < action_horizon`；实际部署还必须满足
  `max_delay_steps <= floor(action_horizon / 2)`，以保证 `d <= H - s`；
- 当前 SpiritAI h=50 的训练 config 新增一个明确命名的 RTC fine-tune variant，而不是
  静默修改既有 config；`max_delay_steps` 在做一次真实端到端 latency profile 后确定。

训练脚本在 `scripts/train.py` 将静态 `config.rtc_training` 传入
`model.compute_loss(...)`。它是 JIT 闭包的一部分，不会作为每个 batch 的 host 参数传递。

### 3.2 Pi0.5 的逐 token time 条件

现有 `Pi0.embed_suffix` 只接收形状为 `(B,)` 的 flow time；Pi0.5 的 AdaRMS 条件也被
广播到整条 action suffix。这不足以表示“前缀已知且干净、后缀仍有噪声”。

改造后 `embed_suffix` 接受 `(B,)` 或 `(B,H)`：

- `(B,)` 仍代表普通训练/采样中的全局 flow time，并在函数内广播为 `(B,H)`；
- `(B,H)` 是 RTC 的每动作 token time；时间 MLP 逐 token 应用，输出 `(B,H,E)`；
- `gemma.RMSNorm` 的 AdaRMS 条件扩展为既接受 `(B,E)` 又接受 `(B,S,E)`。后者将
  scale、shift、gate 与 suffix token 对齐，而不改变任何参数形状或名称。

因此，旧 Pi0.5 参数可以加载：普通路径的每个 token 仍接收完全相同的 time embedding。
新增的是张量广播能力，而不是可训练层。

OpenPI 的时间约定与论文相反：本仓库中 `time=1` 是噪声、`time=0` 是干净动作。
下文所有公式都按本仓库约定书写。

### 3.3 RTC loss

对 batch 内每个样本独立采样 `d ~ Uniform{0, ..., max_delay_steps}` 和一个原有分布的
后缀时间 `t`。令 `P[i] = (i < d)`：

```text
time[i] = 0                         if P[i]
          t                         otherwise
x_t[i]  = A[i]                      if P[i]
          t * epsilon[i] + (1-t)*A[i] otherwise
u[i]    = epsilon[i] - A[i]
loss    = mean_{i: not P[i]} mean_D (v_theta(x_t, observation, time)[i] - u[i])^2
```

也就是说，前缀是 ground-truth、无噪声输入；只对 postfix 求 loss。`d=0` 退化为普通
flow-matching 样本。实现必须按实际 postfix token 数归一化，避免 `d` 大的样本因有效 token
较少而拥有较小权重。`max_delay_steps` 不允许等于 `H`，从而每个样本总有至少一个 postfix
token。

数据加载器无需重排 episode：当前 loader 已在同一时刻提供 observation 与长度 `H` 的真实
action chunk；RTC 的 prefix/postfix 都来自该 chunk。

## 4. Pi0.5 采样与网络接口

### 4.1 模型采样 API

用如下显式参数替换当前实验性的 `rtc_target / rtc_weight / rtc_beta`：

```python
sample_actions(
    rng, observation, *, num_steps=10, noise=None,
    rtc_action_prefix: Array | None = None,  # (B,H,D)，仅 [:d] 有效
    rtc_delay_steps: IntArray | None = None, # (B,)
)
```

两者必须同时出现或同时为 `None`。`delay=0` 合法；`delay >= H`、shape 不符、非 Pi0.5
模型，均在请求进入 JIT 前验证并报错。

若启用 RTC，在每一个 Euler step 的**前**执行：

```text
prefix_mask = arange(H) < delay
x_t          = where(prefix_mask, action_prefix, x_t)
token_time   = where(prefix_mask, 0, scalar_flow_time)
v_t          = model(observation, x_t, token_time)
x_t          = x_t + dt * v_t
```

循环退出后再次固定 prefix，保证浮点积分不会改变它。这与论文 Algorithm 1 等价，只是将
论文“clean=1”的 time 翻译为 OpenPI 的“clean=0”。它是硬 freeze；不再包含 replacement
blend 或任何 VJP。

### 4.2 Policy / WebSocket

`Policy.infer`、`WebsocketClientPolicy` 和 `WebsocketPolicyServer` 继续使用兼容 envelope：

```python
{
  "obs": ...,
  "rtc": {
    "algorithm": "training_time_v1",
    "action_prefix": float32[H,D],
    "delay_steps": int
  },
  "return_model_actions": true
}
```

服务端拒绝未知 algorithm、缺字段及不支持 RTC 的 checkpoint；不再接受 `mask` 或 `beta`。
RTC worker 总是请求 `model_actions`：动作前缀必须在归一化且 pad 至 32 维的模型空间中构造，
不能由 SpiritAI 25 维 robot command 反推。

`create_trained_policy` 自动在 policy metadata 合并以下能力信息（用户自定义 metadata 不被
覆盖）：

```yaml
rtc_capabilities:
  algorithm: training_time_v1       # 或 disabled
  model_type: pi05
  action_horizon: 50
  action_dim: 32
  training_max_delay_steps: <int>
```

客户端在启动时读取并校验 YAML。这样 `action_horizon` 和 `action_dim` 不再需要由人工写进
执行命令。

## 5. 实时执行器

### 5.1 状态与时间线

新增 `openpi.rtc` 下的纯时间线状态和 SpiritAI 运行时适配层，替代当前 `RTCState` + 同步
prefetch。核心实体是：

- `ActionPlan`：`generation_tick`、模型空间 `(H,32)` actions、机器人空间 `(H,C)` actions；
- `InflightRequest`：启动 tick、旧 plan ID、规划延迟 `d_plan`、执行 horizon `s`、冻结前缀；
- `RTCController`：延迟历史、当前 plan、一个 in-flight worker、失败计数与 safe-hold 状态。

控制器把时间离散为 `source_hz` 的 tick。首次同步得到 `A` 后，`A` 在 tick `t0` 逻辑生效。
在 `t0+s` 启动下一次异步请求 `B`，其中：

```text
s = max(d_plan, s_min)
action_prefix[0:d_plan] = A[s : s + d_plan]
```

这正是 RTC 的两块 action chunk 的对齐方式：旧 chunk `A` 在当前推理启动前已运行 `s` 步，
而前缀 `A[s:s+d]` 会在 `B` 生成期间执行。有效性条件为 `d_plan <= H-s`。`B` 在逻辑时间
`t0+s` 开始；结果到达后，`B` 在其已冻结的 prefix 区间与 `A` 完全相同，随后无缝接管。

### 5.2 时延预测、接受和降级

模型无法在请求发出时知道精确结束时间。控制器使用一次真实端到端请求（观测编码、socket、
server、模型、反序列化）记录的 tick 延迟，按照滑动窗口的最大值加上安全 margin 计算
`d_plan`。这是保守上界，而不是事后把错误的 prefix 当作正确 prefix。

- `d_actual <= d_plan`：接受结果；在实际到达前后 prefix 均是旧 plan 的同一动作；
- `d_actual > d_plan`：绝不切换到该结果；记录 deadline miss，旧 plan 继续执行，并在下一个
  有效时机用更新后的延迟预测重新请求；
- `d_plan > training_max_delay_steps` 或 `d_plan > H-s`：不发 RTC 请求并进入显式降级；
- 当前 plan 耗尽而没有安全可用的下一 plan：发送由当前观测构造的单步 hold command；达到
  配置的连续上限后停止循环并报错，要求人工处理。

不会偷偷切换到当前实验性的 soft mask/replacement 路径。

### 5.3 命令发送模型

当前 SpiritAI client 只在 robot idle 时整块发送 command，无法保证在 `t+d` 精确替换仍在执行
chunk。因此 RTC 模式改为**每个控制 tick 最多发送一个 action**：主线程拥有 robot websocket，
按照当前 logical plan 的 action index 发送单步 command；后台线程独占 policy websocket 做一条
inference。这样不要求 robot server 支持 enqueue 或 preempt，仍可在结果一到达时按上述时间线
切换 plan。

现有 rollback suppression、blend 和 motion limit 保留，但在 action 即将发送时以单步形式调用，
并用上一条实际已接受 command 作为历史。`model_actions` 用于模型内 hard prefix；经过输出
transform 和安全限制后的 robot commands 用于物理安全。二者不得相互反推。

启动前只做**只读** RPC preflight，测量 `get_status + get_obs` 的 RTT；不得为测量 RTT 向真实
机器人发送探测 command。运行时将每次真实 `send_command` 的 ack 时间纳入预算统计。若只读
RTT 已耗尽单步预算，或连续实际 command RTT 使预算不可达，RTC client 停止调度并提示降低
`source_hz` 或让 robot server 提供队列接口，而不是表面上运行一个失真的 RTC 时间线。

## 6. YAML 与命令行

训练仍沿用仓库的 Python `TrainConfig` 注册机制；执行器改用单一 YAML profile。新增
`examples/spirit-ai/configs/rtc/training_time.yaml` 作为默认模板，结构如下：

```yaml
schema_version: 1
policy:
  host: localhost
  port: 8000
  prompt: fold the paper box
  connect_timeout_s: 1.0
robot:
  url: ws://172.16.0.30:8766
  action_layout: cartesian
control:
  source_hz: 15.0
  max_steps: 2000
  rpc_budget_fraction: 0.70
  command_ack_timeout_s: 1.0
  robot_idle_timeout_s: 10.0
  motion_limits: {}                 # 复用现有 joint/cartesian 限制字段
rtc:
  mode: training_time
  s_min: 5
  initial_inference_timeout_s: 10.0
  delay:
    planned_max_steps: 12           # 必须 <= metadata.training_max_delay_steps
    history_window: 16
    safety_margin_steps: 1
  deadline_miss:
    max_consecutive: 2
    action: hold_then_stop
```

`planned_max_steps: 12` 只是 h=50 模板的初始上限；首次训练前应根据真实端到端 profile 将
训练 `max_delay_steps` 和部署上限设为同一范围。严格 YAML loader 拒绝未知键、错误类型、
不支持的 `mode`、timeout 非正值和 metadata 不匹配。`policy.connect_timeout_s` 小于
`rtc.initial_inference_timeout_s`，这样一次 socket connect 不会占满整个启动 wait；这只约束
每次 socket attempt，Python 不能强制终止任意系统调用。

入口在模块顶层只定义稳定的默认路径，不读取或解析 YAML：

```python
DEFAULT_RUNTIME_CONFIG = Path(__file__).parent / "configs/rtc/training_time.yaml"

@dataclasses.dataclass(frozen=True)
class BootstrapArgs:
    config: Path = DEFAULT_RUNTIME_CONFIG
    dry_run: bool = False
```

`main()` 在 CLI bootstrap 解析完成后才加载 `BootstrapArgs.config`、校验它并构造完整
`RuntimeConfig`。这避免测试、导入工具或文档生成时意外读取本地文件，也使相对路径不依赖
当前工作目录。启动时必须记录解析后的 config 路径、robot URL、RTC mode 和 `source_hz`。

SpiritAI 的启动入口收敛为：

```bash
uv run examples/spirit-ai/main.py
```

需要替换部署 profile 时使用 `--config /absolute/or/relative/profile.yaml`；`--config` 覆盖的是
默认 profile 路径，而不是把一长串 RTC 参数重新暴露到命令行。只保留 `--config` 与少量运维
覆盖（如日志级别、dry-run）。旧的 `--enable-rtc`、
`--rtc-beta`、`--rtc-s-min`、`--rtc-model-action-*`、`--prefetch-*`、`--execute-steps` 将删除；
它们迁移到 YAML 或由 policy metadata 推导。

## 7. 迁移与代码边界

已重写或移除旧实验性 RTC 表面：

- 删除 `compute_soft_mask`、`build_rtc_target_and_mask` 和旧 `RTCState` 的 replacement 语义；
- 删除 Pi0 的 `rtc_target / rtc_weight` 采样分支及 Policy 对 `mask/beta` 的转发；
- 删除 SpiritAI 的同步 `prefetch_next_chunk` RTC 逻辑；RTC 运行器只使用 YAML 配置的单飞
  asynchronous scheduler，不能与旧 prefetch 语义交叉；
- 新 RTC controller 只由 SpiritAI entrypoint 使用，避免悄悄改变 ALOHA、DROID、LIBERO。

实施保持可 bisect：模型与训练开关、policy/websocket metadata、SpiritAI runtime、YAML 和
non-hardware verification 分别提交；完整任务分解见[实施计划](../plans/2026-08-07-training-time-rtc.md)。

## 8. 验证矩阵与验收

| 层级 | 必测内容 |
|---|---|
| 模型数学 | `d=0` 等价普通损失；prefix 输入始终不变；postfix-only loss 归一化正确；per-token AdaRMS shape 与旧 checkpoint load 正确 |
| API | `rtc_action_prefix/delay` 的 shape、dtype、范围验证；WebSocket envelope 往返；metadata 能力校验 |
| 时间线 | `A[s:s+d]` 正确成为 `B` prefix；`d_actual<=d_plan` 无缝切换；miss 结果绝不采用；horizon 违反进入 hold |
| 配置 | 默认 YAML 路径以 `__file__` 解析、`--config` 可覆盖、YAML 严格解析、训练 config 默认关闭、RTC 只能用于 JAX Pi0.5 |
| SpiritAI 安全 | 单步 command 仍经过现有 blend、rollback guard 与 motion limit；robot ack 拒绝不会推进 logical tick |
| 集成/硬件 dry-run | fake clock + fake policy 延迟；真实 robot 不使能执行的 RPC budget profile；之后才做低速 RTC smoke test |

真实评测至少报告：端到端 inference latency 分布、`d_plan/d_actual`、deadline-miss 数、hold 数、
切换点 command delta、控制频率和任务成功率。训练对照为普通 fine-tune 与 RTC fine-tune，在同一
推理步数、同一 `source_hz`、同一 latency profile 下运行。

## 9. 风险与缓解

- **训练/部署延迟错配：** 先 profile，再把 `training_max_delay_steps` 当作 capability 上报；
  runtime 不允许超过它。
- **单步 RPC 赶不上控制频率：** preflight 拒绝启动；后续若 robot server 支持安全队列，可新增
  adapter，但不改变本期算法接口。
- **JIT 编译或 AdaRMS shape 回归：** 保持 `(B,)` 普通路径可用，使用小型 Pi0.5 fixture 覆盖
  `(B,H)`；将 RTC shape 固定在 `(B,H,D)`，避免编译时的动态长度。
- **模型空间与物理空间混淆：** 只使用 raw `model_actions` 构造 prefix，单独追踪实际 command。
- **失效的旧文档/参数：** 与删除旧 API 同一变更更新 SpiritAI README，明确这是 training-time
  RTC，不宣称 VJP/soft mask 支持。

## 10. 上线前操作检查

1. 使用 `rtc_training.enabled` 的 JAX Pi0.5 checkpoint，确认 policy metadata 为
   `training_time_v1`；
2. 在目标 `source_hz` 下先测量端到端 latency，再同时选择训练 max delay 和 YAML planned delay；
3. 确认 robot 与 policy 都是 `ws://` endpoint，并先使用 `--dry-run` 验证配置、metadata 和
   read-only preflight；
4. 检查 YAML 中 policy connect、initial inference、command ACK 和 robot-idle timeout 是否适合
   当前网络与机器人；
5. 仅在上述条件满足且操作员授权后进行低速硬件 smoke test，记录第 8 节列出的全部指标。
