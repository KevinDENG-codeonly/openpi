# Spirit AI Real Robot Deployment Notes

This note records the current Thor `robot_server` deployment status, real robot inference behavior, and motion-tuning conclusions. Keep the main [`README.md`](README.md) focused on the standard workflow; use this file for operational context and experimental interpretation.

## Current Deployment

- Thor runs `robot_server` in Docker and listens on `ws://172.16.0.30:8766`.
- Precision runs the local policy server on `localhost:8000`.
- [`main.py`](main.py) bridges the local policy server to the remote Thor `robot_server`.
- Current Thor metadata:

| Field | Value |
|-------|-------|
| `structure` | `wholebody` |
| `joint_dim` | `25` |
| `accepted_joint_dims` | `[16, 22, 25]` |
| required cameras | `cam_high`, `cam_left_wrist`, `cam_right_wrist` |

External following must be enabled for real arm motion:

```bash
--enable-external-following
```

Without external following, `robot_server` can accept command chunks while arm joint states do not meaningfully follow.

## Work Completed

- Replaced the old direct SDK/RealSense real robot path with a two-machine bridge.
- Added `robot_server` protocol support for enabling external following mode.
- Added chunk-start smoothing:
  - `--blend-steps`
  - `--rollback-guard-steps`
  - `--rollback-scale`
- Added Precision-side motion limiting:
  - `--max-arm-velocity-rad-s`
  - `--max-torso-velocity-rad-s`
  - `--max-gripper-velocity-s`
  - `--max-base-speed`
  - `--max-joint-accel-rad-s2`
- Added logs for actual state delta, command first-frame delta, raw/limited max velocity, and `limited_fraction`.

## Current Motion Findings

The robot moves reliably with external following enabled, but the latest tests show a tradeoff between less shaking and good continuity.

| Setting | Jitter | Continuity | Task intent | Notes |
|---------|--------|------------|-------------|-------|
| `source_hz=15`, `arm=0.28`, `torso=0.15`, `accel=0.8`, prefetch `0.85` | noticeable shake | medium-low | good | `limited_fraction` often around `0.45-0.60` |
| Same, prefetch `0.1` | worse shake | poor | unstable | prefetch likely too early; stale observation effect |
| Same, `--no-prefetch-next-chunk` | slightly less shake | poor | acceptable | less stale observation, but chunk-to-chunk waiting hurts continuity |

Observed trends:

| Variable | Trend |
|----------|-------|
| Lower velocity | Less shaking, but too low can make motion slow and discontinuous |
| Higher velocity | More responsive, but shaking returns more easily |
| `accel=0.0` | Least shaking in earlier notes, but continuity was worse |
| `accel=0.8-1.2` | Better continuity, slightly more shaking |
| `source_hz=12` | More continuous but weaker task intent |
| `source_hz=15` | Best balance seen so far |
| `source_hz=20` | More responsive but can shake more |

## Interpretation

The remaining issue is probably not one single parameter.

1. Policy output is still high-velocity relative to safe real robot execution; logs often show `raw_max_vel` much larger than `limited_max_vel`.
2. The limiter is changing a large fraction of the output; `limited_fraction` around `0.45-0.60` means the executed trajectory is materially different from the model output.
3. Chunk-level control is fighting continuous motion; each 10-frame chunk is smoothed locally, but there is no global trajectory optimizer across chunks.
4. Prefetch is a tradeoff: early prefetch can use stale observations, while no prefetch adds inference gaps.
5. Thor-side PCHIP interpolation upsamples each chunk to 120Hz, but does not enforce global velocity, acceleration, or jerk continuity across chunk boundaries.

## Safest Reference Command

Use short runs while debugging:

```bash
uv run python examples/spirit-ai/main.py \
  --policy-host localhost \
  --policy-port 8000 \
  --robot-url ws://172.16.0.30:8766 \
  --prompt "Assemble the cardboard box by erecting the flat sheet and folding the side flaps" \
  --enable-external-following \
  --startup-delay-s 10 \
  --source-hz 15 \
  --blend-steps 4 \
  --rollback-guard-steps 4 \
  --rollback-scale 0.2 \
  --max-arm-velocity-rad-s 0.28 \
  --max-torso-velocity-rad-s 0.15 \
  --max-gripper-velocity-s 0.8 \
  --max-base-speed 0.05 \
  --max-joint-accel-rad-s2 0.8 \
  --no-prefetch-next-chunk \
  --max-steps 10
```

Use [`motion_benchmark_plan.md`](motion_benchmark_plan.md) for the full benchmark matrix.

## Recommended Next Engineering Work

Parameter tuning alone is reaching diminishing returns. The next useful implementation work is one of:

- Add a cross-chunk trajectory buffer that blends from the currently executing tail into the next policy chunk.
- Move velocity, acceleration, and jerk limiting to Thor after PCHIP resampling, so the final 120Hz command stream is globally limited.
- Log policy inference latency, chunk idle time, and boundary discontinuity between the previous chunk tail and the next chunk head.
- Replay a recorded human/demo joint trajectory through the same `robot_server` path. If replay is smooth but policy control is not, the issue is mostly policy output and bridge execution. If replay also shakes, the issue is lower in the robot command or interpolation path.
