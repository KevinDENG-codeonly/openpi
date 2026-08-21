# Spirit AI Training-Time RTC Motion Benchmark

Use this benchmark to choose a measured YAML profile for the training-time RTC
runner. It does not use the retired per-run chunk, prefetch, or RTC-guidance flags.

## Preconditions

- Use a JAX Pi0.5 checkpoint trained with `rtc_training.enabled`; its policy metadata
  must advertise `training_time_v1`.
- Use non-TLS `ws://` robot and policy endpoints. The runner rejects `wss://` before
  policy or hardware activity because it cannot enforce the required total write
  deadline on Python TLS sockets.
- Measure end-to-end latency at the intended control frequency before training and
  deployment. The checkpoint training `max_delay_steps` and YAML
  `rtc.delay.planned_max_steps` must use the same measured safe range.
- Start each new profile with dry run:

```bash
uv run examples/spirit-ai/main.py --config PATH --dry-run
```

After dry-run validation and explicit operator authorization, omit `--dry-run` for
one low-speed real-robot run using the same profile.

## YAML profile sweep

Copy `examples/spirit-ai/configs/rtc/training_time.yaml` for each candidate and
change only the fields being evaluated. The primary benchmark fields are:

- `control.source_hz`
- `control.motion_limits.max_arm_velocity_rad_s`
- `control.motion_limits.max_torso_velocity_rad_s`
- `control.motion_limits.max_joint_accel_rad_s2`
- `control.max_steps`

Keep the timeout and transport settings unchanged unless the benchmark specifically
measures them: `policy.connect_timeout_s`, `rtc.initial_inference_timeout_s`,
`control.command_ack_timeout_s`, and `control.robot_idle_timeout_s` are safety
bounds, not throughput knobs.

## What to record

For every profile, record the required runtime metrics and operator observations:

| Run | Profile | dplan | dactual | misses | holds | switch command delta | control frequency | e2e inference latency | jitter 1-5 | task intent 1-5 | notes |
|-----|---------|-------|---------|--------|-------|----------------------|-------------------|-----------------------|------------|-----------------|-------|
|     |         |       |         |        |       |                      |                   |                       |            |                 |       |

Scoring:

- `jitter`: `1` = very shaky, `5` = stable.
- `task intent`: `1` = no useful task progress, `5` = clearly progressing.
- Any deadline miss, hold, command ACK timeout, or RPC-budget stop requires latency
  investigation before increasing control frequency or continuing the sweep.

## Phase 1: Find `control.source_hz`

Keep the motion-limit fields fixed in the YAML candidates. Begin with the measured
training rate and test only frequencies that leave adequate read-only RPC and command
ACK budget. For example, prepare separate profiles at `15.0`, `20.0`, `25.0`, and
`12.0` Hz; reject a candidate if its latency metrics exceed the trained/planned delay
budget or it causes misses/holds.

Do not choose a higher frequency solely because motion appears smoother. The selected
profile must also sustain its measured control frequency and retain a safe command
delta at plan switches.

## Phase 2: Find velocity limits

Fix the selected `control.source_hz`, then compare YAML profiles with conservative,
incremental arm and torso velocity limits. Begin below the known hardware limit and
increase only after a low-speed run is stable. If motion is sluggish, inspect command
delta, control frequency, and latency before relaxing a limit; if jitter returns,
return to the prior profile and investigate the model/data behavior.

## Phase 3: Find acceleration limit

Fix `control.source_hz` and velocity limits, then compare
`control.motion_limits.max_joint_accel_rad_s2` candidates. Keep `0.0` only when the
resulting low-speed motion and switch deltas are acceptable; otherwise prefer the
smallest limit that avoids abrupt starts, stops, or reversals.

## Decision and reporting

Select the YAML profile only when it has no unexplained misses or holds, stays within
the trained/planned delay capability, and is acceptable in the low-speed hardware
run. Archive the exact YAML profile with the metrics table above. Do not compensate
for an unsafe result with retired prefetch, execution-window, or RTC-guidance flags.
