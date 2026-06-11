# Pre-statement skeleton — OM gate run at the proven training regime — DRAFT

**Date.** 2026-06-11.
**Author.** Farhad Safaei.
**Status.** DRAFT SKELETON (Phase 4.1 of the gate-verdict processing). **Nothing here runs.**
Blocked on, in order: the #177 unblock slice (vision head + U1 + U2 per
`OM_UNBLOCK_ASSESSMENT_2026-06-11.md` §3, its own `feat` PR(s) with founder approval), the
[FOUNDER/ADVISOR] estimator decision (consultation ask iii), the [FOUNDER] camera-set /
budget / hardware decisions (§3 below), and **separate explicit founder approval of the
completed pre-statement**. Slots marked [FOUNDER] are deliberately blank.

**Standing clause (immutable):** the recorded AS verdict
(`spikes/results/stage1-AS-2026-06-11/`, exit 5) **stands regardless of anything this run
yields**. This run measures the OM axis; it explains nothing about AS and overturns nothing.

## §1 — What the gate measures (prereg'd protocol, verbatim; nothing new)

Per `spikes/preregistration/OM.yaml` @ `prereg-stage1-OM-2026-05-15` (blob-SHA verified at
launch): conditions `stage1_pickplace_vision_only` (homogeneous) vs
`stage1_pickplace_vision_plus_force_torque_plus_proprio` (heterogeneous), same agent tuple
(panda_wristcam, fetch) in both — **no homo-operationalization confound of the AS kind**;
seeds [0,1,2,3,4]; 20 episodes/seed/condition; cluster bootstrap; gate on
`ci_low_pp ≥ 20` via `chamber-spike next-stage --prior-stage 1` (one axis suffices —
ADR-007 §Validation criteria).

**Estimator [FOUNDER, per consultation ask iii]:** the prereg pins `iqm_success_rate`; the
W-C pinning is now demonstrated on real gate data (AS: IQM CI [−2, 0] vs mean −11 pp). If
the ADR-008 amendment (gate-aggregator choice) lands before this prereg's protocol is
invoked, name the governing aggregator here: [ IQM as-pinned / mean / both-reported,
gate-on-___ ]. No retroactive application to AS.

## §2 — Training regime per cell (condition-symmetric; Rev 17/18 mandates)

Training settings inherit the proven AS regime: N=[FOUNDER: 1024 default, subject to §3
benchmark], γ=0.8, frames=[FOUNDER: 20M default, subject to §3], `rollout_length`/`batch`
per the §3 VRAM benchmark, **PBRS settle α=0.5 ON for both OM conditions** (Rev 18 symmetry;
with the U2 fix mandatory — Φ's placement gate from privileged env state so the vision-only
keep-set cannot skew the shaping), training-time filter off (Rev 17), Rev 15/16 contracts,
prereg YAMLs and `SCHEMA_VERSION` untouched. **No tuning of either condition between
approval and launch; asymmetric adjustments forbidden.**

**Wall-clock honesty (Phase-3 measurement):** the AS regime's 5.5 h/axis does NOT transfer —
OM is rendering-bound (570–810 env steps/s with the current three 128×128 RGB-D cameras,
plateauing in N). At 20M × 10 cells the local floor is O(100+ h) before CNN costs.

## §3 — Pre-launch decisions and their evidence (all [FOUNDER]; none default silently)

1. **Camera set:** keep all three cameras (slowest, no definition question) vs ego wristcam
   only (~3× env throughput; keep-set-class per the filter's documented note — the prereg
   pins modality *families* — but it shapes what "vision" means: ADR-007 note required if
   exercised).
2. **Frame budget:** 20M (AS-proven, state-mode provenance only) vs an OM-specific budget —
   any reduction needs a consolidation pilot (e.g. 1 seed × budget candidates) BEFORE this
   pre-statement freezes; the pilot is its own small pre-statement.
3. **Hardware:** local RTX 2080 (long wall-clock) vs A100 rental (REMEDIATION_LOG §8.4
   reopens for OM; rendering scales with the card).
4. **The §3-of-the-assessment benchmark** (throughput + VRAM at candidate N × camera-set ×
   rollout with the actual vision head) runs after the #177 slice lands and BEFORE this
   document freezes; its numbers replace §2's brackets.

## §4 — Instrument

The prereg'd 20-episode eval governs the gate. Diagnostic sidecars (rider-1 occupancy +
hold-qvel; the W-A/W-B instrument) carried per cell exactly as in the AS gate archive.
Pre-stated watch items: W-A (cold place < 27/30 = α-too-big at OM scale), W-B
(P(placed|static) decomposition), W-C (estimator pinning — resolved by §1's slot), plus
**W-D (new): per-condition rendering-induced regime divergence** — both OM conditions render
identically by construction; the launch log must verify identical camera sets and obs shapes
per cell.

## §5 — Decision rule [FOUNDER draft at freeze]

Skeleton: gate read per §1's estimator on the prereg protocol; exit 0 → Stage 2 unlocks
(P9 tag-cutting becomes available; decided separately); exit 5 → with AS already closed
<20 pp, **both axes would then have fired and closed** — the R-RES-01 both-axes condition
becomes evaluable for the first time (register decision; pre-committed responses: §4a
runbook, ADR-008 fallbacks, consultation discipline).

## Cross-references

`OM_UNBLOCK_ASSESSMENT_2026-06-11.md` (U1/U2 + costs); issue #177;
`spikes/preregistration/OM.yaml` @ tag; ADR-007 §Validation criteria + Revs 9/17/18;
ADR-008 §Decision + the `_spike_next_stage.py` estimator pre-commitment;
`GATE_REGIME_SPEC_2026-06-11.md` (the AS template this mirrors);
`CONSULTATION_BRIEF_GATE_VERDICT_2026-06-11.md` asks (ii)/(iii).
