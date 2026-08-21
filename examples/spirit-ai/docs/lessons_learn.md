# Lessons Learned: Training-Time RTC on Spirit AI Moz1

This document records operating guidance for the current SpiritAI training-time
RTC runner. Earlier replacement-inpainting notes are intentionally removed: they
describe a retired implementation and must not be used to tune a deployed robot.

## Current RTC model and runtime

RTC requires a JAX Pi0.5 checkpoint trained with `rtc_training.enabled`. Its policy
metadata must advertise `rtc_capabilities.algorithm: training_time_v1`; the runtime
uses that metadata, rather than human-entered dimensions, to validate the model
action horizon, action dimension, and maximum trained delay.

Sampling is hard action-prefix conditioning. The next plan receives the already
committed raw model-action prefix while the model generates its postfix. There is
no VJP/PiGDM, `beta`, soft mask, replacement inpainting, or legacy RTC mode to tune.
The prefix remains in model space; one robot-facing action is independently mapped,
limited, and dispatched by the main control thread on each accepted tick.

## YAML-only deployment

Run the source-relative default profile with:

```bash
uv run examples/spirit-ai/main.py --dry-run
```

Use `--config PATH` to select another strict profile. The YAML config owns policy
and robot endpoints, action layout, motion limits, `source_hz`, delay scheduling,
and all timeout values. `--dry-run` suppresses every robot command and is required
for a new profile before an operator-authorized low-speed hardware run.

Do not restore old per-run RTC, prefetch, or chunk-execution flags. The runtime has
one policy inference in flight and one robot command per control tick; it does not
use the former synchronous prefetch loop.

## Transport, latency, and timeouts

The RTC profile requires non-TLS `ws://` robot and policy endpoints. Linux total
write-deadline enforcement depends on `MSG_DONTWAIT` for each socket send; Python
TLS sockets cannot safely provide that guarantee, so `wss://` is rejected before
policy or robot hardware activity.

Measure end-to-end policy and robot latency at the intended `control.source_hz`
before training and deployment. Choose training `max_delay_steps` and runtime
`rtc.delay.planned_max_steps` from the same measured safe range. The default values
are templates, not a latency result.

The profile bounds policy connection attempts (`policy.connect_timeout_s`), bootstrap
and initial inference waits (`rtc.initial_inference_timeout_s`), robot RPC responses,
commands, and writes (`control.command_ack_timeout_s`), and total busy-status waits
(`control.robot_idle_timeout_s`). Timeout, deadline, and RPC-budget failures fail
closed: scheduling stops, transports close, and the configured terminal one-row hold
is dispatched before stopping when that is safe to do.

## Motion safety and diagnostics

Motion limits, blend, and rollback suppression remain part of the YAML safety
profile. Tune them only after dry-run validation and a measured low-speed test; a
large command delta at a plan switch means the physical command is diverging from
the intended plan and should be investigated rather than masked with a retired RTC
parameter.

For every RTC experiment, report:

- `dplan` and `dactual`;
- deadline misses and holds;
- command delta at plan switches;
- achieved control frequency; and
- end-to-end inference latency.

These metrics, together with task outcome and safety-limit observations, are the
comparison basis for ordinary and RTC-trained checkpoints.

## Quick reference: symptom → safe next step

| Symptom | Safe next step |
|---------|----------------|
| Policy metadata rejects the profile | Confirm the checkpoint was trained with `rtc_training.enabled` and the planned delay does not exceed its advertised capability. |
| Deadline misses or repeated holds | Measure latency again; reduce `source_hz` or use a profile whose training and planned delays match the observed system. |
| Robot RPC budget violation | Investigate robot/server/network latency before resuming; do not increase a retired prefetch or guidance setting. |
| Large switch command delta | Review motion limits, blend, rollback suppression, and data/model behavior using a low-speed run. |
| TLS endpoint configured | Change both relevant RTC endpoints to `ws://`; WSS is intentionally rejected for total write-deadline safety. |
