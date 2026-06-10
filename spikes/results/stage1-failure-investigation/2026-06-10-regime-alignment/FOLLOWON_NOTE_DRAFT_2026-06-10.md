# Follow-on note (DRAFT — founder cut) — after the regime-alignment consolidates branch

**Date.** 2026-06-10.
**Author.** Farhad Safaei.
**Status.** DRAFT for founder decision. Nothing in this note is launched or scheduled; it names
the next questions the characterization (`REGIME_ALIGNMENT_CHARACTERIZATION_2026-06-10.md`)
opened, per the pre-statement's consolidates branch.

---

## (a) Place conversion status at scale

**The staged reward's place rung climbed on its own.** At γ=0.8 / N=1,024 / 20M frames, training
`place_rate` reaches ~0.83 and cold placement is 30/30 on all three seeds — **PB1 (PBRS place
shaping) is not needed for placement** and stays unfired (REMEDIATION_LOG §6's "named next
lever" is moot on this evidence).

**The new bottleneck is the `success` conjunct, not placement.** `success = is_obj_placed ∧
is_robot_static` and the static term never consolidates (training success ≤0.0016; cold 0/30 on
A1; A2/A3 logged 2/30 and 1/30 incidentally). Mechanistically plausible and checkable: the
reward grants `static_reward × is_obj_placed` only while placed, and at γ=0.8 the tail value of
holding still may be under-weighted relative to continued reward-bearing motion; alternatively
the policy oscillates around the goal sphere (place flag flickers). Cheapest isolating reads,
in order:

1. **Zero-compute:** from the archived A1 JSONLs + a 5-episode rollout dump, measure whether
   `is_obj_placed` is *sustained* (consecutive-step runs) or flickering at episode end, and the
   `static_rate` trajectory within placed windows. (chamber-analyze over a rollout dump; no new
   training.)
2. **One run:** A1-regime seed 0 with `episode_length` unchanged but eval-horizon analysis at
   H=100 vs H=200 from the same checkpoint (no retraining) — does success appear given more
   settle time? (Mirrors the H3 protocol from the campaign log; loads, does not train.)
3. Only if 1–2 implicate the reward's static term: a PBRS-shaped settle term would be a **new
   lever** requiring its own pre-statement (out of this slice; I1).

## (b) Gate-spike implication

Per the approved Rev 17 note: the gate-scale spike (5 seeds × 2 conditions, A100, 5 GPU-h/axis)
**adopts the aligned regime condition-symmetrically** — N=1,024, 20M frames, γ=0.8,
32k-transition updates, identical on AS-homo and AS-hetero. Measured local throughput (11.1k
steps/s on an RTX 2080 SUPER; 20M ≈ 33 min ≈ 0.5 GPU-h) means the existing A100 budget covers
field-scale sampling several times over — the 10-cell AS axis fits in ≈5 A100-hours with margin.
Two blockers before the gate spike can be specified:

1. **The success-conjunct question above** — a *success*-gap gate cannot clear while success is
   structurally floored on both conditions; either the static term consolidates after (a) is
   understood, or the gate's success definition is examined at ADR level (heavier; not
   proposed).
2. **Training-time safety posture for gate cells** — gate cells at N>1 run filter-off under the
   Rev 17 operator-override record. If the founder wants filtered gate cells instead, the
   `action_linf_component` decision (queue entry) and a batched filter implementation both
   precede the spike.

## (c) Engineering follow-ups surfaced by this slice (for the code PR / issues)

1. **RUNID-COLLISION** (characterization §6): include a config hash (or nonce) in
   `compute_run_metadata`'s `run_id` so same-seed same-commit runs cannot collide; archive
   copies should also carry arm-disambiguated names by default. ADR-002-adjacent (provenance).
2. **`action_linf_component` recalibration** — already queued
   (`DECISION_QUEUE_ENTRY_action_linf_2026-06-10.md`); A3 supplies the isolation evidence.
3. **Filtered-cell wall-clock**: A3 showed the per-step QP solve was ~5× of the old cell's
   wall-clock; any future filtered training at scale needs the batched-QP design or accepts the
   cost knowingly.

## Cross-references

`REGIME_ALIGNMENT_CHARACTERIZATION_2026-06-10.md` (the evidence), `PRESTATEMENT.md` §5
(consolidates branch), `ADR007_REVISION_NOTE_DRAFT.md` (Rev 17), ADR-007 §Stage 1b, ADR-004,
`../2026-06-09-grasp-remediation/REMEDIATION_LOG.md` §6 (the now-moot PB1 ladder entry).
