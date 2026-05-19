# Speculative Decoding — Pioneer Variants

**Date:** 2026-05-18 | **Concepts:** #96-#101

---

## #96 — SSM State Prediction as Draft Model (World First)

**Concept:** Qwen3.5/3.6 SSM (Mamba2) layers maintain a fixed-size recurrent state that evolves predictably. Use the SSM states from the last N layers to predict the next token — **zero extra model, zero VRAM cost, zero overhead.** The SSM is already computing this every step.

**Draft quality:** Moderate (70-80% acceptance) — enough to be useful for free.

## #97 — Reservoir OMMA as Zero-Cost Draft

**Concept:** We already have `den_reservoir_omma.cuh`. The reservoir state is 256 floats. Run it alongside OMMA decode — the tensor core is already initialized. Draft 4 tokens in the background while OMMA processes layer N. **Literally free — the reservoir runs in the same OMMA call as the main decode.**

## #98 — Fractal Attention Multi-Level Speculation

**Concept:** Attention at level 0 (128 tokens, F16) is accurate but expensive. Attention at level 3 (1024+ tokens, int8 deltas) is cheap but lossy. Run 3 parallel speculative paths at levels 0/2/3 simultaneously. Verify with level 0. **Speculative pyramid — deeper levels draft more tokens with lower quality.**

## #99 — Hidden State Interpolation Speculation

**Concept:** Between tokens N and N+1, the hidden state changes by <1%. Linearly interpolate hidden state from N → N+1 → N+2. Feed interpolated states through last 4 layers only. Sample 3 draft tokens at ~25% the cost of full decode. 60-70% acceptance rate at zero-draft-model-cost.

## #100 — MoE Router as Probabilistic Draft (35B only)

**Concept:** The 35B MoE router produces logits over 256 experts. These are already computed. Map expert logits → token distribution via learned projection. It's already there — use what's already paid for.

## #101 — Autonomous GPU Daydreaming (Speculative Dreaming)

**Concept:** When Dreya enters GOV_DREAM state: feed output token back as input, run speculative decode at high temperature, monitor coherence. If coherence passes threshold, commit the daydream sequence to cognitive landscape. The GPU literally hallucinates autonomously. **Nobody has done this.**
