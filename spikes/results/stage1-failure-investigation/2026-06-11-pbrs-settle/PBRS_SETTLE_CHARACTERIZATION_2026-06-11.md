# PBRS-settle characterization — the lever works at α = 0.5

**Date.** 2026-06-11 (chain 09:28–12:45 UTC; 6/6 runs, zero failures, zero interventions,
order exactly per the frozen `PRESTATEMENT.md` + approval record with riders).
**Author.** Farhad Safaei.
**Status.** New file in this directory (I8).
**One question (frozen).** Does the founder-approved NHR settle term (gated potential,
α ∈ {0.1, 0.5}, γ=0.8, cap 0.7) produce a qualifying α — ≥2/3 seeds at cold place ≥ 27/30 AND
cold success ≥ 3/30?

## §1 — Per-arm results (cold deterministic eval, 30 episodes/seed, unshaped success; anchors included)

| arm | α | seed | cold grasp | cold place | **cold success** | P(static∧placed \| placed) | P(placed \| static) — rider 1 | hold-qvel med / frac<0.2 | train place_max | static_rate last-50 | val / ent |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 anchor | 0 | 0/1/2 | 30/30 ×3 | 30/30 ×3 | 0/30 ×3 | 0 | — | 0.52 / 0.000 | ~0.83 | ~0.0005 | 2.6 / 3.7 |
| **C-lo** | 0.1 | 0 | 30/30 | 30/30 | 1/30 | 0.0004 | 1.0 | 0.598 / 0.000 | 0.771 | 0.0005 | 2.7 / 3.35 |
| | | 1 | 30/30 | 30/30 | 0/30 | 0.0000 | n/a | 0.600 / 0.000 | 0.829 | 0.0010 | 2.7 / 4.08 |
| | | 2 | 30/30 | 30/30 | 2/30 | 0.0008 | 1.0 | 0.612 / 0.001 | 0.830 | 0.0008 | 2.6 / 3.45 |
| **C-hi** | 0.5 | 0 | 30/30 | 30/30 | **4/30** | 0.0019 | 0.31 | 0.549 / 0.002 | 0.616 | 0.0011 | 2.7 / 3.80 |
| | | 1 | 30/30 | 30/30 | **11/30** | 0.0062 | 1.0 | 0.502 / 0.006 | 0.574 | 0.0007 | 2.6 / 3.58 |
| | | 2 | 30/30 | 30/30 | 0/30 | 0.0000 | n/a | 0.488 / 0.000 | 0.546 | 0.0013 | 2.7 / 4.12 |

Run ids (all distinct; #214 fingerprint vintage): C-lo `cf6db9934f617fdf` / `e97eda81debac69e`
/ `836702d80233d04f`; C-hi `c84990eff663def9` / `1c13765858f38cad` / `49186fe3d99e8de6`.
Code vintage: `feat/pbrs-settle-shaping` @ `b910a2b` (PR #219), the established pre-merge
run pattern.

## §2 — Verdict, strictly per the frozen rule

- **C-lo (α=0.1): does not qualify.** Place bar passes 3/3 (30/30 each; no regression — the
  minimal-interference design held), but success 1/0/2 is under the 3/30 bar on every seed.
  α=0.1 under-drives, as its framing anticipated.
- **C-hi (α=0.5): QUALIFIES.** Place bar passes 3/3 (30/30 each — cold place did not regress
  at the assertive α). Success bar: seed 0 **4/30** ✓, seed 1 **11/30** ✓, seed 2 0/30 ✗ —
  **2/3 seeds**, exactly the frozen bar.

**The lever works: α\* = 0.5.** No tie-break needed (one qualifying α). Per the frozen rule
the next artifact is the **gate-regime spec draft at (γ=0.8, N=1024, 20M, PBRS α\*=0.5)** —
Phase 4a, drafted only after this verdict is reported to the founder (stop-point discipline);
**the gate spike launches only on separate, explicit founder approval.**

This is the **first qualifying success result in the entire campaign**: every prior cell
(broken-init triplet, fix-only triplet, regime-alignment A1/A2/A3, γ-scan ×7) sat at 0–2/30.

## §3 — Reads beyond the bar (recorded for the gate spec and the record)

1. **The mechanism did what the theorem said and only that.** Cold place stayed 30/30 on all
   six shaped runs (the α-too-big signature never fired at eval), value/entropy stayed at the
   healthy γ=0.8 profile (val ≈ 2.7 vs A2's pathological 51), and success emerged exactly
   where the temporal-credit analysis predicted — at the assertive α, via deliberate
   late-episode settling (C-hi hold-qvel medians 0.49–0.55, lower tail fattening:
   frac<0.2 = 0.002–0.006 vs 0.000 unshaped).
2. **Success remains seed-variable (4 / 11 / 0).** The bar is met but not saturated; the gate
   protocol's 5 seeds × 20 episodes will sample this honestly. The spread is the main
   uncertainty the gate spec must carry, and argues for reporting per-seed ladders in the
   gate archive, as throughout this campaign.
3. **Rider 1 earned its place.** On C-hi seed 0, P(placed | static) = 0.31 — two-thirds of
   its static steps occur with placement lost (the drift-on-settle mode the pre-flight's
   1-of-6 nuance predicted); seed 1, by contrast, holds P(placed | static) = 1.0 and converts
   11/30. Settling *while keeping the cube in the sphere* is the remaining skill gradient —
   visible only because the joint-occupancy instrument was mandated.
4. **Training-side place_max declined with α** (unshaped ~0.83 → C-lo 0.77–0.83 → C-hi
   0.55–0.62) while cold place held at 30/30. The stochastic training rollouts trade some
   in-episode place occupancy for settling attempts; the deterministic policy keeps both.
   Watch item for the gate spec, not a blocker (the gate measures cold success).
5. **Rider 2 (ungated variant) stays unfired** — the works-branch makes its conditional
   (consultation-brief mention) moot.

## §4 — Standing notes

PA1/PA2 verdicts remain suspended under their two recorded confounds (unchanged). The
PBRS-settle shaping is training-cell-only; every number in §1's success columns is the
**unshaped** prereg'd predicate.

## §5 — Artifacts (this directory; `SHA256SUMS.txt` covers all)

Per run: `{tag}_results.json` (cold ladder + rider-1 occupancy fields + hold-qvel margins),
`{tag}_{run_id}.jsonl`, `{tag}_eval_steps.jsonl`, `launch_{tag}_trimmed.log`; plus
`chain_timeline.log`, `SETTLE_REACHABILITY_PROBE_2026-06-11.md`, `PRESTATEMENT.md` (frozen),
`PRESTATEMENT_APPROVAL_2026-06-11.md` (riders), `ADR_NOTE_DRAFT.md` (Rev 18; landed via
PR #219).

## Cross-references

`PRESTATEMENT.md` §Decision rule (the frozen verdict source); PR #219 (the shaping code +
ADR-007 Rev 18); `../2026-06-11-gamma-scan/` (no-γ\* verdict; design margins);
`../2026-06-10-success-static-probe/` (C1); `../2026-06-10-regime-alignment/` (A1 anchors);
REMEDIATION_LOG §2 (PA1's lesson honoured) + §6 (this ladder entry, now resolved);
Ng, Harada & Russell (ICML 1999).
