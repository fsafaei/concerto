# EXPLORATORY homo-static characterization — the moving partner does NOT explain the reversal

**Date.** 2026-06-11 (chain 20:55–22:21 UTC; 3/3 runs, zero failures, single launch SHA
`6f99d75` across all cells — the no-repository-mutations rule held).
**Author.** Farhad Safaei.

**EXPLORATORY — binding labels:** this slice is **outside the prereg'd AS conditions**. It is
not gate evidence; it cannot change, re-run, or re-frame the recorded AS verdict
(`spikes/results/stage1-AS-2026-06-11/`, **exit 5 — immutable**). Its output is
interpretation evidence for the consultation only.

**One question (frozen).** Does making the homo partner static move homo settle-conversion
into the hetero band — isolating the #230 mis-port's perpetual partner motion as the
explanation for the gate's sign reversal?

## §1 — Results against the gate reference bands

X-homo-static: AS-homo embodiment, zero-action partner (training AND eval), gate regime
verbatim (N=1024, γ=0.8, 20M, α=0.5, filter-off); cold eval 30 episodes/seed, unshaped,
unfiltered, prereg'd init, W-B occupancy instrument.

| cell | cold grasp | cold place | **cold success** | P(static∧placed \| placed) | P(placed \| static) | hold-qvel med / frac<0.2 | train place_max | val / ent |
|---|---|---|---|---|---|---|---|---|
| X-homo-static s0 (`3fcfe84ac1b1f559`) | 30/30 | 30/30 | **0/30** | 0.0 | n/a (never static) | 0.524 / 0.000 | 0.453 | 2.7 / 3.52 |
| X-homo-static s1 (`73f6142b112779d1`) | 30/30 | 30/30 | **0/30** | 0.0 | n/a | 0.462 / 0.000 | 0.551 | 2.7 / 3.93 |
| X-homo-static s2 (`54b2878225d158fc`) | 30/30 | 30/30 | **0/30** | 0.0 | n/a | 0.578 / 0.000 | 0.645 | 2.7 / 4.02 |
| *gate homo-as-run (reference)* | — | 30/30 ×5 (replay) | **1/100** (0,0,0,1,0) | ≤0.0012 | 1.0 where defined | ~0.5–0.6 | 0.55–0.62 | 2.7 / ~3.8 |
| *gate hetero (reference)* | — | 30/30 ×5 (replay) | **12/100** (3,7,0,2,0) | up to 0.0062 | 0.29–1.0 | 0.49–0.55, tails fattened | 0.55–0.62 | 2.7 / ~3.8 |

**Pooled X-homo-static success: 0/90.**

## §2 — Verdict, strictly per the frozen rule (§5 of the pre-statement)

- *Moving-partner explains* required pooled ≥ 12 % (≥ 11/90) with ≥ 2/3 seeds nonzero:
  **not met** (0/90; 0/3 seeds nonzero).
- *Does not explain* fires at the homo-as-run floor (≤ 2/90): **0/90 is at the floor.**

**Verdict: DOES NOT EXPLAIN.** Freezing the homo partner — which removes both pre-stated
disturbance paths (physical perturbation AND observation-vector non-stationarity) and both
pre-stated timings (learning-time and eval-time), per the approval record's mechanism notes
— did not move homo settle-conversion at all. The sign reversal does not rest on the
partner's *motion*. Per the pre-stated limit and rule text, the reversal now rests on
**embodiment/crowding-as-genuine-signal or variance/selection-pressure**, and the
consultation carries that; this cell cannot separate arm-*presence* from
selection-pressure asymmetry.

## §3 — Reads beyond the bar (recorded; interpretation-only)

1. **The homo deficit is conversion-attempt suppression, fully intact under a static
   partner.** All three cells: zero static steps in 9,000 placed-step-bearing eval steps
   (`frac<0.2 = 0.000`; hold-qvel medians 0.46–0.58, the familiar band; P(static∧placed |
   placed) = 0). The hetero cells' fattened lower tails (frac<0.2 up to 0.006) never appear
   on homo, moving or static. Whatever suppresses homo settling survives total partner
   stillness.
2. **Everything upstream of settling is healthy and condition-comparable:** grasp+place
   30/30 on all cells; PPO health at the clean γ=0.8 profile; training place_max 0.45–0.65
   (the shaped-cell band).
3. **What this strengthens, stated carefully:** reading (b)'s *presence* variant (a second
   7-DOF arm in reach of the workspace suppresses settle attempts even when motionless —
   plausibly via the ego's static observation of it or learned spatial caution) and/or the
   selection-pressure leg of reading (a) (α\*=0.5 was selected on hetero; nothing tuned the
   settle incentive against the homo observation/geometry). It *weakens* the mis-port's
   motion as the operative confound — #230's classification question loses urgency for the
   *verdict interpretation* (the bug is real and still wants fixing, but fixing it would not
   have changed this outcome on this evidence).

## §4 — Artifacts (this directory; `SHA256SUMS.txt` covers all)

`PRESTATEMENT.md` (frozen) + `PRESTATEMENT_APPROVAL_2026-06-11.md` (the timing addendum);
per-seed `x-homo-static_seed{N}_results.json` (EXPLORATORY arm strings; W-B fields inline),
`x-homo-static_seed{N}_{run_id}.jsonl` (every line stamped
`exploratory_partner_static: true`), `x-homo-static_seed{N}_eval_steps.jsonl`,
`launch_*_trimmed.log`, `chain_timeline.log` (single SHA).

## Cross-references

`spikes/results/stage1-AS-2026-06-11/` (the immutable verdict + reference bands);
`../2026-06-11-gate-verdict/` (operationalization audit; consultation brief §2 — this
characterization feeds asks i/ii); issues #230 (classification question — see §3.3), #233;
PR #234 (the structurally-gated knob); ADR-007 Revs 12, 15–18; ADR-002; ADR-009.
