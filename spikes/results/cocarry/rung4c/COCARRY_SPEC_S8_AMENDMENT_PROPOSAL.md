# Co-carry spec §8 amendment proposal — embodiment-invariant stress measure

**Status:** proposal for the founder to fold into the co-carry spec
(R-2026-06-B §8, the success-predicate stress conjunct + §15 f_max). The spec
lives in the planning kit (not in this repo), so this records the amendment
here. Grounded in the Rung-4b instrument diagnosis and the Rung-4c resolution
(`spikes/results/cocarry/rung4b`, `…/rung4c`). Companion to the §4 amendment
(rigid → calibrated compliant drive) under `rung4b/`.

## What §8 currently says (wrist-force stress proxy)

The success predicate's `is_unstressed` conjunct (and the training reward's
excess-stress penalty) read the **wrist constraint-solver force proxy**: the
linear-force norm of each holding arm's wrist-link incoming joint force
(`get_link_incoming_joint_forces()` at `panda_hand` / the partner's wrist),
max over holding arms, gated at f_max = 130.6 N (derived from the matched-Panda
distribution, §15).

## Why amend it

The wrist incoming-joint force folds in the **arm's own link weight/inertia** —
mass that never reaches the bar — so it is **embodiment-dependent**: a
different-bodied partner reads a different wrist force for the *same* bar state.
Rung-4b measured the xArm6 (Robotiq) wrist at 138 N (p50) / 514 N (max) just
holding the bar still at a zero-action hold, vs the Panda's 40 / 82 N — over the
Panda-calibrated f_max before any cooperation. A Panda-calibrated f_max applied
to this proxy rejects a different body on a measurement artifact, not on a
cooperation failure. This blocks any valid cross-embodiment (EH) measurement.

## Proposed amended §8 text

> **§8 Stress conjunct (embodiment-invariant coupling force).** For the
> compliant-coupling task variant (§4), the `is_unstressed` conjunct and the
> reward excess-stress penalty read the **bar coupling force** — the force the
> dual-hold spring transmits, `F = K_c · ‖grip_frame_world − bar_end_world‖`,
> max over the holding arms (capped at the drive force-limit). This is a
> function of the bar/coupling **geometry** only, so it is the **same physical
> quantity regardless of which arm body holds the bar** (embodiment-invariant),
> unlike the wrist incoming-joint-force proxy (which folds in the arm's own
> link dynamics). Implemented as `chamber.envs.cocarry` `stress_measure`
> (`"wrist"` = the rigid-ladder proxy, default & byte-identical; `"coupling"` =
> this measure). **f_max is re-derived per stress measure** from the
> matched-Panda-pair distribution (§15): for the coupling measure,
> f_max_coupling = 1.25 × matched-success coupling p99 = **366 N**, then held.
> The rigid ladder (Rungs 0–4) keeps the wrist proxy + 130.6 N (immutable, I8).

## Companion finding (for §15 / the EH result)

Under this invariant measure the embodiment effect is **real, not an
artifact**: against a fixed cooperative ego, swapping the Panda partner → xArm6
takes the coupling load from ~267 N (in-band) to ~1867 N (~5× f_max, ~7× the
matched pair), failing `unstressed` 12/12, while every same-body control-style
(policy-shift) teammate stays in-band (267–334 N). Embodiment heterogeneity
degrades cooperation via coupling over-stress, ~6× beyond the control-style
spread. (The frozen rigid-incumbent does not transfer to the compliant
coupling, so the incumbent-specific EH Δ awaits a compliant-coupling re-freeze;
the effect is ego-independent.)

_Refs: ADR-026 §Decision 4 + §Open-questions (stress-measure resolution);
ADR-005; R-2026-06-B §8/§15 Rung 4b/4c; `spikes/results/cocarry/rung4b`,
`…/rung4c`._
