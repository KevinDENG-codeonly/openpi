# Spirit AI Real Robot Operational Notes

These notes describe the current training-time RTC runner for the Thor
`robot_server` bridge. The former chunk-prefetch and flag-driven deployment guide
is retired and must not be used for robot operation.

## Current deployment and profile

- Thor runs `robot_server` and the default profile uses its non-TLS `ws://` robot
  endpoint.
- Precision runs the policy server; its effective RTC policy transport must also be
  non-TLS `ws://`. The runner rejects `wss://` before policy or hardware activity
  because Python TLS sockets cannot provide the Linux total write-deadline guarantee.
- `examples/spirit-ai/main.py` owns the robot WebSocket. Its single-flight worker
  owns the policy connection, and the main thread sends at most one robot action per
  accepted control tick.
- The JAX Pi0.5 checkpoint must have been trained with `rtc_training.enabled` and
  advertise `rtc_capabilities.algorithm: training_time_v1`.

Choose the deployment settings in a strict YAML profile, based on
`examples/spirit-ai/configs/rtc/training_time.yaml`:

- `robot.action_layout` must be `joint` for a 27D joint checkpoint or `cartesian`
  for a 25D Cartesian checkpoint. Serve the matching checkpoint; do not try to adapt
  a mismatched action layout at runtime.
- Set `robot.enable_external_following: true` before an authorized real-arm run.
  Without it, the server may accept commands while arm joint state does not
  meaningfully follow.
- Configure blend, rollback, and motion safety under `control.blend_steps`,
  `control.rollback_guard_steps`, `control.rollback_scale`, and
  `control.motion_limits`.
- Keep `policy.connect_timeout_s`, `rtc.initial_inference_timeout_s`,
  `control.command_ack_timeout_s`, and `control.robot_idle_timeout_s` as explicit
  safety bounds rather than throughput-tuning controls.

## Safe launch sequence

Use the canonical dry-run command for every new profile or checkpoint:

```bash
uv run examples/spirit-ai/main.py --config PATH --dry-run
```

Confirm the resolved YAML path, checkpoint capability metadata, action layout, and
read-only robot preflight. Measure end-to-end latency at the selected
`control.source_hz` before training and deployment, then keep training
`max_delay_steps` and YAML `rtc.delay.planned_max_steps` in the same safe range.
Only after explicit operator authorization should a low-speed run omit `--dry-run`.

## Required runtime evidence

For every hardware experiment, record:

- `dplan` and `dactual`;
- deadline misses and holds;
- command delta at plan switches;
- achieved control frequency; and
- end-to-end inference latency.

Investigate a command ACK timeout, RPC-budget stop, deadline miss, hold, or large
switch delta before continuing. The runner stops scheduling and closes transports on
configured failures; when applicable it sends the configured one-row terminal hold
before stopping.

## Historical physical observations

The following observations are retained only as historical context from pre-
training-time-RTC experiments, not as a tuning prescription: lower velocity limits
often reduced visible shaking but could slow task progress, while higher limits made
the system more responsive but could reveal jitter. Re-evaluate these tradeoffs with
the YAML benchmark process in [`motion_benchmark_plan.md`](motion_benchmark_plan.md)
and the required runtime metrics above.
