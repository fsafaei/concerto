# γ-scan characterization — the discount lever does not bridge the static valley

**Date.** 2026-06-11 (chain 04:21–08:39 UTC; 7/7 runs, zero failures, zero interventions,
order exactly per the frozen `PRESTATEMENT.md` + `PRESTATEMENT_APPROVAL_2026-06-11.md`).
**Author.** Farhad Safaei.
**Status.** New file in this directory (I8).
**One question (frozen).** Is there a gate-viable γ\* — ≥2/3 seeds at cold place ≥ 27/30 AND
cold success ≥ 3/30 — between the measured extremes?

## §1 — Per-arm results (cold deterministic eval, 30 episodes/seed; anchors included)

| arm | γ | frames | seed | cold grasp | cold place | cold success | P(static∧placed \| placed) | hold-qvel med / p10 / min | train place_max | static_rate last-50 | value / entropy / adv_std |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 anchor | 0.80 | 20M | 0/1/2 | 30/30 ×3 | 30/30 ×3 | 0/30 ×3 | 0 (probe: 0/5,637) | 0.521 / 0.483 / 0.317 | ~0.83 | ~0.0005 | 2.6 / 3.7 / 0.09 |
| **B** | 0.80 | **40M** | 0 | 30/30 | 30/30 | **0/30** | **0.0** | **0.663 / 0.625 / 0.603** | 0.862 | **0.0001** | 2.7 / 3.91 / 0.06 |
| **A-0.9** | 0.90 | 20M | 0 | 30/30 | 30/30 | 0/30 | 0.0 | 0.563 / 0.503 / 0.400 | 0.848 | 0.0002 | 5.5 / 4.57 / 0.20 |
| | | | 1 | 30/30 | 30/30 | 0/30 | 0.0 | 0.602 / 0.552 / 0.315 | 0.802 | 0.0005 | 5.4 / 4.68 / 0.22 |
| | | | 2 | 30/30 | 30/30 | 1/30 | 0.0004 | 0.613 / 0.589 / 0.146 | 0.789 | 0.0002 | 5.3 / 4.45 / 0.23 |
| **A-0.95** | 0.95 | 20M | 0 | 30/30 | **0/30** | 0/30 | n/a (never placed) | n/a | **0.012** | 0.0298 | 9.0 / 3.87 / 0.45 |
| | | | 1 | 30/30 | 30/30 | 0/30 | 0.0 | 0.526 / 0.495 / 0.202 | 0.751 | 0.0003 | 10.9 / 4.88 / 0.54 |
| | | | 2 | 30/30 | **2/30** | 1/30 | 0.0099 | 0.487 / 0.267 / 0.166 | **0.014** | 0.0126 | 8.6 / 3.65 / 0.47 |
| A2 anchor | 0.99 | 20M | 0 | 30/30 | 16/30 | 2/30 | (2/16 episodes) | — | 0.098 | 0.0008 | 51.1 / 5.08 / 2.47 |

Run ids: A-0.9 `5b5f84522e6fc768` / `b5cfb7f01e3eda58` / `3f58bae9b163dbac`; A-0.95
`7b62e797df704bbe` / `8807ca68889a6eaf` / `d2e32c689de38930`; B `2801bb2688106342` — all
distinct (the #214 config-fingerprint fix in production; first archive at the new vintage).

## §2 — Verdict, strictly per the frozen rule

- **A-0.9:** place bar passes 3/3 (30/30 each); success bar **fails 0/3** (0, 0, 1 vs ≥3/30
  required). Does not qualify.
- **A-0.95:** place bar **fails 2/3** (0/30, 30/30, 2/30); success 0, 0, 1. Does not qualify.
- **B (γ=0.8, 40M):** success 0/30 — the B-alone budget clause does **not** fire; the
  declining-static_rate prediction is confirmed at 2× budget.

**No arm qualifies. The no-γ\* branch fires:** per the frozen rule, the scan's measured
margins become the design-evidence base for option C, and the option-C pre-statement is
**drafted only** (Phase 4b; nothing launches). Drafting awaits the post-verdict founder
report per the stop-point discipline.

## §3 — The γ-trade-off read (why there is no sweet spot)

1. **The static valley is never crossed at any place-consolidating γ — and optimisation moves
   *away* from it.** Hold-speed medians: 0.52 (γ=0.8, 20M) → **0.66 (γ=0.8, 40M)** → 0.56–0.61
   (γ=0.9) → 0.49–0.53 (γ=0.95, the one placing seed). Every placing run holds at 2.4–3.3×
   the 0.2 rad/s threshold with `P(static∧placed | placed) ≤ 0.01`, and B shows the hold
   *accelerating* with budget — training at low γ optimises deeper into the moving-hold
   equilibrium, exactly as the probe's near-zero-gradient-valley mechanism predicts
   (`1 − tanh(5·qvel)` ≈ 0.01 at these speeds; the paying region starts ≈ 0.35 rad/s away).
2. **Place consolidation collapses before stillness pays.** At γ=0.95, 2/3 seeds fail to
   consolidate place *in training at all* (place_max 0.012 / 0.014 — not an eval-transfer
   gap), with the A2 pathology signature already onset (value 8.6–10.9 vs 5.4 at γ=0.9 and
   2.7 at γ=0.8; adv_std 0.45–0.54). The trade is not a smooth frontier with a crossing
   point: place reliability degrades (seed-fragility) at γ ≈ 0.95 while the success rung is
   still at the floor; by γ=0.99 (A2) place is down to 16/30 episodes for 2/16 successes.
3. **The isolated γ=0.99 successes look like stochastic luck, not a learnable basin.** Across
   the whole campaign success has fired cold 5 times total (A2 2, A3 1, A-0.9 s2 1,
   A-0.95 s2 1) — always 1–2 episodes, never repeating within a seed, with
   `P(static∧placed | placed)` ≤ 0.01 even in those runs. Nothing on the discount axis
   converts that into the ≥3/30-on-≥2/3-seeds reliability the rule demands.
4. **What the margins give option C (the design evidence the rule asked for):** the gap to
   close is ≈ 0.3–0.45 rad/s of hold speed (median 0.49–0.66 vs threshold 0.2); per-step
   gradients must be non-negligible across exactly that band (the PBRS potential must not
   plateau before ≈ 0.7 rad/s); and γ can stay at the place-friendly 0.8 — the shaping term,
   not the discount, carries the stillness incentive. p10/min columns show the distribution's
   lower tail already brushes the threshold occasionally (min 0.146–0.32 on several runs):
   the behaviour is reachable, it is the *incentive density* that is missing — the
   textbook PBRS case.

## §4 — PA-suspension note (standing; no re-verdicting)

Unchanged from the regime-alignment characterization §4: PA1/PA2 verdicts remain suspended
under their two recorded confounds.

## §5 — Artifacts (this directory; `SHA256SUMS.txt` covers all)

Per run: `{tag}_results.json` (cold ladder + `P(static∧placed | placed)` + hold-qvel margins
inline), `{tag}_{run_id}.jsonl` (per-window training stream), `{tag}_eval_steps.jsonl`
(per-step cold-eval instrument), `launch_{tag}_trimmed.log`; plus `chain_timeline.log`,
`PRESTATEMENT.md` (frozen, byte-stable since approval), `PRESTATEMENT_APPROVAL_2026-06-11.md`.

## Cross-references

`PRESTATEMENT.md` §Decision rule (the frozen verdict source);
`../2026-06-10-success-static-probe/` (C1 verdict + options note — option C is the named
next step, founder cut); `../2026-06-10-regime-alignment/` (A1/A2 anchors); ADR-007 §Stage 1b
Rev 17; ADR-002 §Revision history 2026-06-10 (run_id fingerprint, first production archive);
PR #217; REMEDIATION_LOG §2 (PA1's non-PBRS lesson — binding on any option-C design).
