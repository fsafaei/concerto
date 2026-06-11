# ADR-007 §Stage 1b — draft revision-history entry (Rev 18): PBRS settle term (training-cell shaping)

**Status.** DRAFT — not landed in `adr/ADR-007-*.md`; approval co-requisite with
`PRESTATEMENT.md` (this directory). Lands with the shaping `feat` PR once approved.
**Author.** Farhad Safaei.

---

## Proposed entry text (for ADR-007 §Revision history)

**Rev 18 (2026-06-11) — potential-based settle shaping in the training cell (P1.05.11;
first reward-side addition since the grasp-remediation campaign closed).** The evidence chain
(Rev 15/16 rig fixes → clamp isolation → Rev 17 regime alignment → C1 probe → γ-scan no-γ\*
verdict → settle-reachability pre-flight) localised the last Stage-1b consolidation gap to a
temporal-credit defect: the `is_static` conjunct of `success` is one zero-delta control step
away from any measured hold state (floor ≈ 0.003 rad/s vs threshold 0.2), but the static
stage term's gradient is near-zero across the 0.2–0.7 rad/s hold band and no
place-consolidating discount re-profiles it (γ-scan, 2026-06-11). This revision authorises a
**potential-based settle term in the training cell**:

1. **Form (exact Ng–Harada–Russell).** F(s, s′) = γ·Φ(s′) − Φ(s) with γ = 0.8 (the training
   MDP's discount — the invariance theorem is stated for the MDP's own γ), and
   Φ(s) = −α · min(‖qvel_arm‖_max, 0.7) · 1[is_obj_placed(s)] — state-only (the placement
   gate is a state predicate; the qvel reduction is the `is_static` predicate's own max over
   the 7 ego arm joints). **Policy invariance:** by NHR 1999 Thm 1, for any state-only Φ the
   optimal policy of the shaped MDP equals the optimal policy of the unshaped MDP; further,
   the discounted shaping sum telescopes to −Φ(s₀) + lim γ^T Φ(s_T) independent of the
   trajectory, so the lever changes the *temporal credit profile* (front-loading ≈ 0.5α onto
   the deceleration transition) and provably nothing else. **PA1's pathology cannot recur by
   construction:** PA1's reach→grasp bridge was an additive non-potential term — it changed
   the optimum and created the deceptive low-value attractor (REMEDIATION_LOG §2); a
   potential-based term cannot relocate any optimum, deceptive or otherwise.
2. **Boundary conventions.** Φ ≡ 0 at true termination (the terminal transition contributes
   F = −Φ(s_T)); at time-limit truncation Φ(s′) is evaluated on the actual final observation
   (the vectorised path's `info["final_observation"]`; the single-env path's pre-reset obs),
   and the trainer's Pardo-style truncation bootstrap operates on the shaped-reward MDP
   unchanged. No Φ-zeroing at truncation.
3. **Placement and gating.** The shaping is a **config-gated training-cell reward transform**
   (`shaping.settle_alpha`, default 0 = off → byte-identical pre-existing behaviour;
   ADR-002). `Stage1PickPlaceEnv.compute_normalized_dense_reward` and `evaluate()` are
   **byte-untouched** — the canonical env reward remains upstream-matched, the success
   predicate unmodified, and all evaluation (cold instruments and the prereg'd gate protocol)
   measures unshaped success. The training-time-only posture mirrors Rev 17's safety-stack
   record: what the cell optimises is cell-internal; what the gate measures is prereg'd.
4. **Condition symmetry (gate-facing mandate, extending Rev 17's clause).** Anything
   gate-facing trains both AS conditions with the identical transform and identical α. The
   2026-06-11 PBRS-settle slice (AS-hetero only, α ∈ {0.1, 0.5} × 3 seeds, pre-statement
   frozen at launch) is investigation, not gate evidence.
5. **Calibration provenance.** α values derived in the pre-statement from measured constants
   (hold band 0.40–0.66 rad/s; one-step settle; floor 0.003; reward scale after /5
   normalisation), with the α-too-big signature (place regression) pre-stated and
   instrumented. cap = 0.7 rad/s from the γ-scan hold-qvel margins.
6. **Discipline.** No prereg YAML/tag edits; no `SCHEMA_VERSION` change (P10); the prereg'd
   gate protocol (condition_ids, 5 seeds, 20 episodes/seed/condition) governs the gate
   unchanged. ADR-007 stays **Accepted**; no §Decision content change.

## Why a revision entry and not a new ADR

The Rev 15/16/17 precedent covers implementation-details-of-the-cell; a reward-side addition
is heavier — it is evidence-surface-changing even when optimum-preserving — which is exactly
why it arrives as an ADR-noted, config-gated, default-off, theorem-backed term with its own
frozen pre-statement, rather than as a silent training tweak. It alters no prereg-locked
comparison surface: the gate's metric, protocol, env reward, and predicate are all untouched.
A *non*-potential reward change, a predicate/threshold change, or a success-definition change
would NOT fit this revision and would require ADR-level review (the options note and the
readiness note both bar them from the default menu).

## Evidence trail

`spikes/results/stage1-failure-investigation/2026-06-11-pbrs-settle/`
(SETTLE_REACHABILITY_PROBE_2026-06-11.md; PRESTATEMENT.md — frozen at launch);
`../2026-06-11-gamma-scan/GAMMA_SCAN_CHARACTERIZATION_2026-06-11.md` (no-γ\* verdict, margins);
`../2026-06-10-success-static-probe/` (C1 mechanism); `../2026-06-10-regime-alignment/`
(A1 anchors); REMEDIATION_LOG §2 (PA1's non-PBRS failure) + §6 (this ladder entry);
Ng, Harada & Russell, "Policy invariance under reward transformations" (ICML 1999).
