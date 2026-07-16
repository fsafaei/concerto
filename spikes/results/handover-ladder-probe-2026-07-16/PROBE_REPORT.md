# Handover ladder Slice 0 — oracle-headroom probe report (2026-07-16)

**Verdict, as computed by the pre-stated rule: NO-GO.** Neither pre-stated
condition holds on either canonical cell: headroom (oracle − REF) = **0.000** on
both cells (bound: ≥ +0.15 on ≥ 1 cell) and learnable-structure (oracle − best
fixed policy) = **0.000** on both cells (bound: ≥ +0.05 on ≥ 1 cell). Eval-only,
CPU-only, **non-gating** (I1): this informs the v1.1 learned-ladder scope
decision only; no leaderboard row changes and no gate claim is made.

Rule-before-result record: `PROBE_PRESTATEMENT.md`, committed before any probe
episode ran (tag `probe-handover-ladder-slice0-2026-07-16`; the branch history
is checkable). Driver: `scripts/repro/_handover_ladder_slice0_probe.py`; the run
is deterministic (P6/ADR-002) and rewrites `probe_results.json` byte-identically
(re-run verified, SHA-256
`9c6fceb63e3a5bfeb1c6afde462330a9db27d2a0e7faac6d4564c3ca263dc206`). Wall time
~2 s on CPU.

## Reconciliation (hard gate) — PASSED

The recomputed REF (the actual `ScriptedHandoverEgo` driven live over the
committed draws) matches the SHA-verified committed bundle
`spikes/results/benchmark/handover-v1/ref-script-2026-07-06/` **per-episode,
500/500** (success, seating force, angular residual, failure mode), and the
recomputed summary equals the committed row exactly: mean 0.338 / IQM 0.176
(`compute_summary`, 2000 resamples, bootstrap root seed 0).

## Per-cell table (same draws, same success predicate; n = 250/cell)

| cell | oracle | REF | always-regrasp | never-regrasp | scripted rule | oracle − REF | oracle − best fixed |
|---|---|---|---|---|---|---|---|
| `presenter_mismatch_30` | 0.548 | 0.548 | 0.548 | 0.500 | 0.548 | **0.000** | **0.000** |
| `presenter_mismatch_45` | 0.128 | 0.128 | 0.128 | 0.048 | 0.128 | **0.000** | **0.000** |

The oracle searched the full ego phase-1 action space per state through the
real env step: analytic translate optimum (exact lateral cancellation; offsets
~3·10⁻⁴ m vs a 0.10 m range) × an 11-point `reorient_deg` grid × both
`regrasp_flag` values. Reorient invariance held on **every** episode (the env
clips the wrist correction analytically; `resolve_placement` treats the ego's
`reorient_deg` as advisory), so the entire reachable outcome set per state is
{no-regrasp, request-regrasp} — and the scripted rule already picks the best
element of that set in every drawn state.

## Infeasibility decomposition — REF's number is the task's ceiling

| cell | feasible-and-found | infeasible-by-budget | infeasible-by-window |
|---|---|---|---|
| `presenter_mismatch_30` | 137 (0.548) | 113 (0.452) | 0 |
| `presenter_mismatch_45` | 32 (0.128) | 218 (0.872) | 0 |

Every single oracle failure is **budget-infeasible by construction**: the
angular window is unreachable by wrist alone (`|err| ≥ 30°` vs wrist 25° +
window 5°) while the re-grasp is blocked by the takt budget (re-grasp needs
skew ≤ 0.07 s; the presenters draw skew ~ max(0, N(0.2 s, 0.1 s)), so ~90% of
draws are blocked). Zero failures are window-infeasible (all draws sit far
inside the 170° re-acquire range — the free-re-grasp counterfactual succeeds on
every failed episode). **REF-SCRIPT's committed 0.176 IQM / 0.338 mean is the
achievable ceiling of these cells, not the scripted rule's shortfall.**

Mechanism note (answers the §1 pointed question in the negative): at the anchor
parameters the seating-force conjunct implies an effective angular threshold of
75 N / 7.5 N·deg⁻¹ = **10°** at zero lateral residual — *looser* than the 5°
angular window — so the scripted rule's regrasp trigger (post-wrist residual ≥
5°) sits exactly at the binding threshold and does not under-request regrasp.
The success set per state is `(|err| < 30°) ∨ (skew ≤ 0.07 s)`, which the
scripted rule attains everywhere; the only state-dependence (regrasp iff the
wrist can't finish) is already a one-threshold rule on an observed quantity.

## Secondary (reported, not gating)

Seating-force proxy distributions are essentially identical at oracle vs REF —
mismatch_30: mean/p50/p90 = 47.4/25.6/132.5 N (oracle) vs 47.6/25.6/132.5 N
(REF); mismatch_45: 137.4/138.1/245.0 N (both) — as expected when the outcome
sets coincide; the oracle's tie-break toward lower force on
executed-regrasp successes accounts for the 0.2 N mean difference on
mismatch_30.

## What NO-GO means downstream

The learned ladder on `handover_place` at the committed anchor cells has
nothing to discriminate: a trained ego cannot beat the scripted reference
(headroom 0), and no fixed-vs-learned separation exists (structure 0). The
never-regrasp floor (0.500/0.048) shows the regrasp decision *matters*, but one
scripted threshold already captures it entirely. Slices 1–4 (training env,
trainer/eval enablement, configs, campaign) should **not** be built on these
cells. The pivot question (co_hold_secure vs a contact-rich handover variant)
moves to the planning-kit decision memo
(`04_engineering/HANDOVER_LADDER_SLICE0_DECISION_2026-07-16.md`); the B-BLIND
definability finding (sensor-blinding, not partner-blinding — no own-channel /
partner-channel split exists in this task's observation) is recorded there for
the founder's disposition.

Files: `PROBE_PRESTATEMENT.md` (the rule), `probe_results.json` (per-cell +
per-episode records + decomposition), `REPRO.txt`, `SHA256SUMS.txt`. Archive
immutable once committed (I8).
