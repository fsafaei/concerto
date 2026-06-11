# Pre-statement skeleton — EXPLORATORY homo cell, partner-motion asymmetry removed — DRAFT

**Date.** 2026-06-11.
**Author.** Farhad Safaei.
**Status.** DRAFT SKELETON (Phase 4.2 of the gate-verdict processing). **Nothing here runs**;
launch requires explicit founder approval of the completed pre-statement.

**EXPLORATORY — the labelling discipline, stated first and repeated in every artifact this
run would produce:** this cell is **outside the prereg'd AS conditions**. It is not a gate
cell, it is not gate evidence, it cannot change or re-run the recorded AS verdict
(`spikes/results/stage1-AS-2026-06-11/`, exit 5 — **immutable**), and its only purpose is to
inform the *interpretation* of that verdict (consultation §2, reading (a) vs (b)). Every
results JSON carries `"EXPLORATORY"` in its arm string; the characterization opens with this
clause; no artifact may frame it as a gate re-measurement.

## §1 — The adapted design (conditioned on Phase 1's finding)

The founder's original framing ("train the homo cell with the identical algorithm/partner
setup as hetero") is **already what ran** — `AS_HOMO_OPERATIONALIZATION_2026-06-11.md` §1
shows both gate conditions trained the same ego-AHT HAPPO learner against the same frozen
scripted partner class. The variable that actually differs is **partner behaviour**, via the
P1.05.10 mis-port acting asymmetrically: the hetero fetch partner is an empirical statue;
the homo panda partner's arm moves through the shared workspace all episode (§2.1 of the
operationalization doc, measured).

**The clean single-variable contrast therefore is:** the recorded homo cells (moving partner
arm) vs an exploratory homo cell with the **partner's motion removed** — a zero-action
`panda_partner` held at the Rev 16 ready pose, the established Rev 13 A1-ablation pattern
(an eval-and-training rig knob, stated as such; partner class/freeze contract otherwise
untouched). This equalises the homo partner's *behaviour* to the hetero partner's empirical
behaviour (static) while keeping the homo *embodiment* (a second 7-DOF arm whose reach
covers the workspace).

What each outcome would mean (pre-stated, interpretation-only):

- **Static-partner homo success rises toward hetero levels** → the recorded homo floor is
  substantially the §2(a) operationalization artifact (the moving-arm perturbation);
  embodiment per se is not shown to penalise settling.
- **Static-partner homo success stays floored** → the moving arm is exonerated; the residual
  candidates are the arm's mere *presence* in the workspace (reading (b), strengthened) or
  the selection-pressure asymmetry (settings tuned on hetero). The cell cannot separate
  those two — pre-stated limit.

## §2 — Arms (skeleton; frozen at founder sign-off)

| arm | partner behaviour | seeds | regime |
|---|---|---|---|
| **X-homo-static** | zero-action panda_partner (ready pose, motionless) | 0/1/2 | the AS gate regime verbatim: N=1024, γ=0.8, 20M, α=0.5, filter-off |

Anchors (no new runs): the recorded gate homo cells (moving partner; 0/0/0/1/0 of 20) and
gate hetero cells. Budget: 3 × ~33 min ≈ **1.7 h** chained, halt-without-retry, no mid-run
changes; zero new engineering (the zero-action partner exists as the Rev 13 ablation
pattern — [FOUNDER: confirm whether to reuse the A1 wrapper or pass a zero-action partner
spec; either is a rig knob, neither touches the partner registry contract]).

## §3 — Instrument

The investigation instrument (NOT the gate's): cold 30-episode deterministic eval per seed,
unshaped success, with the rider-1 occupancy sidecars and hold-qvel margins — directly
comparable to the gate cells' W-A/W-B replay columns. Training-side ladders + PPO health as
in every campaign archive. Collision-proof run_ids; new dated directory; I8; SHA256SUMS.

## §4 — Decision rule [FOUNDER draft at freeze; interpretation-only by construction]

Skeleton: pre-state the comparison thresholds before launch (e.g. "X-homo-static pooled cold
success ≥ K/90 ⇒ the moving-arm confound is established as material"), with the explicit
clause that **neither branch alters the recorded verdict, the prereg, the register, or any
gate artifact** — the output feeds the consultation reply and, if warranted, a future
freshly-preregistered AS protocol (its own ADR-level discussion, not this document's).

## Cross-references

`AS_HOMO_OPERATIONALIZATION_2026-06-11.md` (the finding this adapts to);
`CONSULTATION_BRIEF_GATE_VERDICT_2026-06-11.md` §2 + ask (ii);
`spikes/results/stage1-AS-2026-06-11/GATE_VERDICT_REPORT_2026-06-11.md` (the immutable
verdict); ADR-007 Rev 13 (the zero-action ablation precedent) + Revs 17/18 (regime);
issue P1.05.10 (the mis-port; its eventual fix is a separate rig slice and would NOT be
applied retroactively to any recorded archive).
