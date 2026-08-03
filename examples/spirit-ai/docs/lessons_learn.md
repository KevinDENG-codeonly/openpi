# Lessons Learned: Real-Time Inference on Spirit AI Moz1

This document captures practical insights from deploying pi05 VLA models on the Spirit AI Moz1 humanoid robot with Real-Time Chunking (RTC). It is intended to help anyone reproducing or extending this work understand the tradeoffs involved, even without prior experience on this platform.

---

## Background

The pi05 model predicts actions in chunks — a single forward pass outputs 50 timesteps of future actions (the "action horizon"). The robot then executes these actions open-loop before re-observing and replanning. This creates a fundamental tension: longer chunks give the model more temporal context, but the robot is "blind" during execution and cannot react to unexpected environment changes until the next inference call.

Our system runs on the following setup:
- **Model**: pi05_spiritai_cart_lora_h50 (action_horizon=50, action_dim=32, Cartesian control)
- **Training data**: collected at 30Hz
- **Inference latency**: ~82ms end-to-end client round-trip (4ms model forward pass; the rest is image preprocessing and network transfer)
- **Robot control**: 500Hz servo loop with PCHIP interpolation from the policy's waypoint rate up to 120Hz

---

## Real-Time Chunking (RTC)

### The Problem It Solves

Without RTC, each action chunk is sampled independently. Even if consecutive chunks start from the same observation, stochasticity in the flow matching process means their predictions can disagree at the boundary — leading to visible jitter, directional reversals, and oscillation. This gets worse as `execute_steps` decreases (more frequent replanning = more boundaries).

### How It Works

RTC modifies the flow matching denoising loop (which iteratively refines noise into a clean action trajectory). After each Euler integration step, RTC blends the intermediate result toward a "target" — the portion of the previous chunk that hasn't been executed yet, shifted to align with the new chunk's timeline. The blending strength is controlled by a soft mask that is strongest for the already-committed region and decays toward zero at the tail of the chunk.

The key parameters are:
- **`rtc_beta`** (0 to 1): Global blending strength. Higher values force the new chunk to stay closer to the previous one. Setting this to 0.9 almost eliminates boundary jitter but makes the model slower to react to environment changes.
- **`rtc_s_min`**: Minimum number of unconstrained (free) steps at the tail of each chunk. These steps have zero mask weight, giving the model full freedom to plan new behavior. Larger values make the model more responsive but reduce smoothness guarantees.

In practice, `rtc_beta=0.9` with `rtc_s_min=5` provides strong inter-chunk consistency while still allowing the model to adapt within ~170ms (5 steps at 30Hz).

### Implementation Note: JAX JIT Compatibility

The `rtc_beta` parameter must be multiplied into the mask outside of the JIT-traced function. If passed as a raw float kwarg into `sample_actions`, JAX will trace it and raise a `TracerBoolConversionError` when switching between RTC-enabled and RTC-disabled calls. The fix is to precompute `rtc_weight = clip(beta * mask)[..., None]` in numpy before passing it to the model as a static-shape array.

---

## Velocity Limiting and the `limited_fraction` Metric

### What It Means

The system applies per-step velocity capping before sending commands to the robot. `limited_fraction` reports what proportion of steps in each chunk had their velocity reduced. For example, `limited_fraction=0.5` means half of the predicted actions were moving faster than allowed and got clamped.

### Why It Matters

High `limited_fraction` (above ~0.3) indicates that the executed trajectory is materially different from what the model predicted. This causes a feedback loop:
1. The model predicts a trajectory to reach position X in 50 steps.
2. Velocity clamping slows the motion, so the robot only reaches position Y after 50 steps.
3. On the next observation, the model sees it hasn't arrived and predicts an aggressive correction.
4. The correction gets clamped again → persistent oscillation / jitter.

Additionally, velocity clamping is applied per-dimension. If only some joints exceed the limit, the coordinated multi-joint trajectory gets distorted (e.g., a straight-line Cartesian motion becomes curved).

### Tuning Guidance

The goal is to keep `limited_fraction` below 0.15. In our testing, the following limits achieve this while remaining safe for the hardware:

```
--max-cart-translation-m-s 0.10
--max-cart-rotation-rad-s 0.30
--max-torso-cart-translation-m-s 0.05
--max-torso-cart-rotation-rad-s 0.20
```

If `limited_fraction` is consistently high, the correct fix is to raise velocity limits — not to lower `source_hz`, which would create a mismatch with the training data timescale.

---

## Choosing `source_hz`

`source_hz` defines the temporal spacing between the policy's output waypoints. The robot-side interpolator upsamples these to 120Hz for smooth servo execution. This parameter should match the training data sampling rate (30Hz in our case) because the model learned its dynamics at that timescale. Setting `source_hz` too high makes the model extrapolate between timesteps it never saw during training; too low makes it skip frames, producing larger per-step deltas that are more likely to hit velocity limits.

With RTC enabled, `source_hz` can be set more aggressively than without, because RTC handles the chunk-boundary discontinuities that previously limited higher rates. However, exceeding 2× the training rate is not recommended.

---

## Prefetch and Observation Freshness

The main inference loop uses prefetching to overlap computation with execution:

1. A chunk of `execute_steps` actions is sent to the robot (e.g., 5 steps = 167ms at 30Hz).
2. Partway through execution (controlled by `prefetch_delay_fraction`), a new observation is captured and inference begins.
3. By the time the current chunk finishes, the next chunk is already computed and can be sent immediately.

This means the effective observation delay is not `execute_steps / source_hz` (167ms) but approximately `execute_steps / source_hz × prefetch_delay_fraction` (~83ms with fraction=0.5). Reducing `execute_steps` below 3 provides marginal improvement in freshness but increases chunk-boundary artifacts and gives the velocity limiter less trajectory to work with.

---

## Anti-Jitter Mechanisms

Beyond RTC, the system provides several mechanisms to suppress motion artifacts:

- **Blend steps** (`--blend-steps 4`): The first N steps of each chunk are linearly interpolated between the robot's current state and the predicted trajectory. This eliminates instantaneous jumps at chunk start.
- **Rollback guard** (`--rollback-guard-steps 6`, `--rollback-scale 0.1`): Detects when the first N steps of a new chunk move backward relative to the previous chunk's direction, and suppresses that motion to 10% of its original magnitude. This prevents the oscillation pattern where the model alternates between "overshoot" and "correct back."

---

## Quick Reference: Symptom → Action

| Symptom | Likely cause | Adjustment |
|---------|-------------|------------|
| Fine jitter / vibration | Velocity limits too tight, or weak RTC | Raise velocity limits until `limited_fraction` < 0.15; raise `rtc_beta` toward 0.9 |
| Sluggish or fails to reach target | Velocity limits too tight, or RTC over-constraining | Raise velocity limits; lower `rtc_beta`; increase `execute_steps` |
| Reversal at chunk boundaries | Model predicts conflicting directions across chunks | Lower `rollback-scale` (e.g., 0.1); raise `rollback-guard-steps` |
| Model ignores environment changes | RTC too strong or `execute_steps` too large | Lower `rtc_beta`; raise `rtc_s_min`; reduce `execute_steps` |
| `limited_fraction` consistently > 0.3 | Velocity caps too conservative | Raise `max-cart-*` limits (do NOT lower `source_hz`) |
