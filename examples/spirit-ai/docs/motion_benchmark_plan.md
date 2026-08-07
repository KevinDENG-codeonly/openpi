# Spirit AI Motion Smoothness Benchmark

This benchmark is for quickly finding a practical balance between low jitter and good action continuity during real robot inference.

## Goal

Tune these runtime parameters:

- `--source-hz`
- `--max-arm-velocity-rad-s`
- `--max-torso-velocity-rad-s`
- `--max-joint-accel-rad-s2`

Use short real robot runs first. Start every candidate with `--max-steps 5`; only increase to `20` after the robot motion looks safe.

## What To Record

For each run, record:

| Run | source_hz | arm_vel | torso_vel | accel | jitter 1-5 | continuity 1-5 | task intent 1-5 | avg limited_fraction | notes |
|-----|-----------|---------|-----------|-------|------------|----------------|-----------------|----------------------|-------|
|     |           |         |           |       |            |                |                 |                      |       |

Scoring:

- `jitter`: `1` = very shaky, `5` = stable.
- `continuity`: `1` = stop-and-go, `5` = smooth continuous motion.
- `task intent`: `1` = no useful task progress, `5` = clearly progressing.

Log interpretation:

- `limited_fraction > 0.7`: limiter is changing most of the policy output; motion may become too slow or off-distribution.
- `limited_fraction 0.2-0.6`: usually a useful range.
- `limited_fraction < 0.1` but motion still shakes: jitter is probably not caused by the bridge velocity limiter.

## Fixed Baseline Command

Use this as the command template. Only change the benchmark parameters shown in each phase.

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
  --prefetch-next-chunk \
  --prefetch-delay-fraction 0.85 \
  --max-steps 5
```

## Phase 1: Find `source_hz`

Keep all other parameters fixed:

```text
arm_vel=0.28
torso_vel=0.15
accel=0.8
```

Test in this order:

| Candidate | Change |
|-----------|--------|
| A | `--source-hz 15` |
| B | `--source-hz 20` |
| C | `--source-hz 25` |
| D | `--source-hz 12` |

Decision rule:

- If `20` is more continuous than `15` without bringing back jitter, use `20`.
- If `25` looks rushed or produces more chunk-boundary jerk, reject `25`.
- If `12` is stable but too slow or task intent drops, reject `12`.

Expected best candidate: `source_hz=20`.

## Phase 2: Find Velocity Limits

Fix the best `source_hz` from Phase 1. Test:

| Candidate | arm_vel | torso_vel | Expected behavior |
|-----------|---------|-----------|-------------------|
| A | `0.22` | `0.12` | Most stable, may be too slow |
| B | `0.28` | `0.15` | Recommended balance |
| C | `0.35` | `0.20` | More responsive, may shake slightly |
| D | `0.45` | `0.25` | Fastest, higher risk |

Decision rule:

- If motion is stable but sluggish, increase one level.
- If task intent is good but jitter returns, decrease one level.
- If `limited_fraction` stays near `1.0`, the velocity limit is too strict for the current policy output.

Expected best candidate: `arm_vel=0.28`, `torso_vel=0.15`, or one level higher if motion is too slow.

## Phase 3: Find Acceleration Limit

Fix the best `source_hz`, `arm_vel`, and `torso_vel` from Phases 1-2. Test:

| Candidate | accel | Expected behavior |
|-----------|-------|-------------------|
| A | `0.0` | No acceleration limiting; fastest response |
| B | `0.5` | Softest, may lag |
| C | `0.8` | Recommended balance |
| D | `1.2` | More responsive, less smoothing |

Decision rule:

- If starts/stops or direction changes feel abrupt, reduce accel.
- If the robot lags behind the intended motion, increase accel or set it to `0.0`.
- If disabling accel improves continuity without jitter, keep `0.0`.

Expected best candidate: `accel=0.8` or `1.2`.

## Suggested Fast Sweep

Run only these six candidates first:

| Run | source_hz | arm_vel | torso_vel | accel |
|-----|-----------|---------|-----------|-------|
| 1 | `15` | `0.28` | `0.15` | `0.8` |
| 2 | `20` | `0.28` | `0.15` | `0.8` |
| 3 | `25` | `0.28` | `0.15` | `0.8` |
| 4 | best from 1-3 | `0.35` | `0.20` | `0.8` |
| 5 | best from 1-3 | `0.22` | `0.12` | `0.8` |
| 6 | best from 1-5 | best from 1-5 | best from 1-5 | `1.2` |

After selecting the best candidate, rerun it with:

```bash
--max-steps 20
```

Only move to longer runs after the 20-step run remains stable and task intent is good.

## Current Recommended Candidate

Start with:

```text
source_hz=20
max_arm_velocity_rad_s=0.28
max_torso_velocity_rad_s=0.15
max_joint_accel_rad_s2=0.8
```

If it is still not continuous enough, try:

```text
source_hz=20
max_arm_velocity_rad_s=0.35
max_torso_velocity_rad_s=0.20
max_joint_accel_rad_s2=1.2
```

If jitter returns, go back to the previous candidate.
