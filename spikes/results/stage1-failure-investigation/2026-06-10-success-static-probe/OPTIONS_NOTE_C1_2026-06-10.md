# Options note — C1 (incentive/discount) on the success-static conjunct

**Date.** 2026-06-10.
**Author.** Farhad Safaei.
**Status.** DRAFT — options only, comparative per I10; founder cut; **nothing here executes**.
Conditioned on `SUCCESS_STATIC_PROBE_2026-06-10.md`'s C1-pure verdict.
**Standing rule, stated up front:** changes to the success definition (the predicate, the 0.2
static threshold, the 0.025 goal_thresh, or a hold/hysteresis variant) are **gate-metric
changes and are not on this menu** — any such change is ADR-level and is not proposed by
default. The options below change *how the policy is trained*, never *what the gate measures*.

## The decision the options serve

The AS gate measures the homo−hetero **success** gap. With success structurally floored on both
conditions (C1), the gate cannot produce a meaningful gap in either direction — so resolving C1
is the critical path to a gate spike, and the choice below sets the gate's training regime.

## Option A — revisit the gate-regime γ with eyes open

Evidence rows already in hand: γ=0.8 → place 30/30 cold but success 0/90 (stillness worth ≈1.0
discounted, unlearned); γ=0.99 → success appears (2/16, 1/1 conditional) but place reliability
collapses (16/30 cold at scale; A2). The middle — **γ ∈ {0.9, 0.95}** — is untested: at γ=0.9
the static tail is worth ≈2, at γ=0.95 ≈4, while the effective horizon stays short enough that
the place-consolidation benefit of low γ may survive. **Cost:** one seed per candidate γ at the
aligned regime ≈ 33 min each; a 2-γ × 1-seed scan ≈ 70 min, then a 3-seed confirm of the winner
≈ 100 min. **Requires:** a fresh pre-statement (new arm class; frozen rule on cold success and
cold place jointly); condition-symmetric if anything gate-facing. **Risk:** the trade may not
have a sweet spot — A2's pathology signature (value 51, entropy 5.1) may reappear by γ=0.95.

## Option B — longer training at γ=0.8 (does static arrive at 40M?)

Cheapest single probe of "is it slow, not absent": one seed, 40M frames ≈ 66 min. **The
measured trend argues against it** — static_rate *declines* monotonically over training on all
three A1 seeds (0.0125 → 0.0005) and the hold-qvel sits at a stable 0.52 rad/s plateau with
zero sub-threshold excursions in 7,530 placed steps; there is no gradient pathway visible in
the data for stillness to emerge from at γ=0.8. **Value:** if run and null, it converts "the
incentive analysis says no" into "measured at 2× budget, no" — closing the cheap branch before
Option C's lever. **Requires:** pre-statement amendment (an A1-extension arm), trivial budget.

## Option C — PBRS settle term (a NEW lever; own pre-statement; ADR-bearing)

A potential-based shaping term over a stillness potential (e.g. φ(s) = −α·tanh(5‖qvel_arm‖)
active while placed), strictly in **Ng–Harada–Russell form** (γφ(s′) − φ(s)) per PA1's lesson —
additive non-PBRS shaping drove the deceptive-optimum collapse and is barred. PBRS provably
preserves the optimal policy, so the gate's optimum is untouched while the gradient toward
stillness is densified at any γ. **Cost:** implementation + 3-seed validation ≈ half a day
including the pre-statement. **Requires:** its own pre-statement AND an ADR-007 revision note
(first reward-side change since the campaign; even optimum-preserving shaping is
evidence-surface-changing); condition-symmetric mandatorily. **This is the first true lever
proposed since the campaign closed — it fires only by founder decision after A/B evidence.**

## Comparative summary

| option | what it tests | cost | new lever? | pre-statement | main risk |
|---|---|---|---|---|---|
| A: γ scan {0.9, 0.95} | the place↔static γ trade has a sweet spot | ~3 h total | no (regime knob, Rev 17 class) | fresh, frozen rule | no sweet spot exists |
| B: 40M @ γ=0.8 | static is slow, not absent | ~66 min | no | amendment | trend says null; spends a slot |
| C: PBRS settle | densify the stillness gradient at any γ | ~half day | **yes** | own + ADR note | shaping-tuning rabbit hole |

A natural sequencing the founder may adopt or reorder: **A first** (regime-knob only, no lever,
directly informs the gate-regime spec), B alongside or skipped on the trend evidence, C only if
A finds no viable γ.

## Dead branches (for the record)

- **C2 framing** (success-difference-as-cooperation-signal): refuted for this conjunct — the
  frozen-partner control is indistinguishable from live, so there is no partner-mediated
  success signal to measure here. The gate's cooperation sensitivity rests on the place/grasp
  chain, not on static.
- **C3 defect-fix slice:** no defect — implementation matches documented panda-routed intent
  exactly (probe step 2).
- **C4 hold/hysteresis:** no flicker measured (single ~190-step placed runs); and any "placed
  for k steps" redefinition would be a gate-metric change — ADR-level, not proposed.
- **C5 horizon extension:** refuted by measurement (H=200 adds zero successes; first place at
  step ~11); no Rev 15-class horizon slice is warranted.

## Cross-references

`SUCCESS_STATIC_PROBE_2026-06-10.md` (the evidence); `../2026-06-10-regime-alignment/`
(characterization §2–§3; PRESTATEMENT §5); ADR-007 §Stage 1b Rev 17; REMEDIATION_LOG §2 (PA1's
non-PBRS lesson); ADR-008 (gate aggregate convention, untouched).
