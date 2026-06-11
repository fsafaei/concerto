# Gate-regime specification — Stage-1b AS spike at the evidenced regime

**Date.** 2026-06-11.
**Author.** Farhad Safaei.
**Status.** DRAFT — Phase 4a of the PBRS-settle slice, drafted on founder instruction after
the works-branch verdict (`../2026-06-11-pbrs-settle/PBRS_SETTLE_CHARACTERIZATION_2026-06-11.md`
§2: α\* = 0.5). **The gate spike launches only on separate, explicit founder approval of this
document — never from the document itself.**

## §1 — What the gate measures (prereg'd protocol, verbatim; nothing here is new)

Per `spikes/preregistration/AS.yaml` (git tag `prereg-stage1-AS-2026-05-15`, blob-SHA-verified
at launch by `chamber-spike verify-prereg`):

- **Conditions:** `stage1_pickplace_panda_only_mappo_shared_param` (homogeneous) vs
  `stage1_pickplace_panda_plus_fetch_ego_aht_happo_per_agent` (heterogeneous).
- **Protocol:** seeds **[0, 1, 2, 3, 4]**, **20 episodes per (seed, condition)** = 100
  episodes per condition. Estimator `iqm_success_rate`; cluster bootstrap;
  `failure_policy: strict`; `run_purpose: leaderboard`.
- **The gate:** ADR-007 §Validation criteria — Stage 2 unlocks iff ≥1 of {AS, OM} shows a
  ≥20 pp homo−hetero gap on the **95 % cluster-bootstrap CI lower bound**. Measured by
  `chamber-spike next-stage --prior-stage 1`: exit 0 iff ≥1 axis at `ci_low_pp ≥ 20.0` (IQM);
  exit 5 otherwise.
- The campaign's 30-episode cold instrument was the investigation's; **the prereg instrument
  governs the gate.** OM is out of scope (#177 blocks it independently; one axis suffices).

## §2 — Training regime per cell (condition-symmetric; FROZEN as evidenced)

Identical for both conditions, exactly the evidenced configuration — **no further tuning of
either condition happens between spec approval and gate launch; asymmetric adjustments are
forbidden** (founder addition 3; Rev 17/18 condition-symmetry clauses):

| setting | value | provenance |
|---|---|---|
| num_envs | 1,024 | Rev 17; regime-alignment characterization |
| total_frames | 20,000,000 per cell | regime-alignment; PBRS-settle arms |
| γ | 0.8 | γ-scan (place-friendly; no-γ\* elsewhere) |
| **shaping** | **PBRS settle, α\* = 0.5, cap 0.7, ON for BOTH conditions** | Rev 18; PBRS-settle characterization §2 |
| training-time safety | filter-off (operator-override record) | Rev 17; D-034 open and non-blocking |
| rollout / batch / hidden / lr | 32 per env (32,768-transition updates) / 4,096 / 256 / 3e-4 | Rev 14/17 lineage |
| episode_length / reset | 100 / Rev 16 ready pose | Rev 15/16 |
| env reward / `evaluate()` / prereg YAMLs / `SCHEMA_VERSION` | byte-untouched | standing constraints |

**Honest-sampling statement (founder addition 3):** hetero cold success is seed-variable at
this regime (4/30, 11/30, 0/30 on seeds 0/1/2). The prereg'd 5-seed × 20-episode protocol
samples that variance honestly, **whatever it yields** — including a sub-bar gap.

## §3 — Preconditions checklist (all must hold at launch)

| # | precondition | status at drafting |
|---|---|---|
| P1 | PR #219 (shaping + Rev 18) merged to `main` | OPEN — blocker until merged |
| P2 | PR #220 (PBRS-settle evidence) merged | OPEN — blocker (the α\* provenance must be on `main`) |
| P3 | #215 dispatch-order fix merged + Tier-2 dispatch test green | ✅ merged (#217, `b5d15da`); test green on `main` |
| P4 | #214 run_id fingerprint merged (collision-proof 10-cell archive) | ✅ merged (#217); verified in production across 13 runs |
| P5 | **AS-homo path smoke at the gate regime** (founder addition 1) | ✅ PASS — `HOMO_SMOKE_2026-06-11.md`: production dispatch end-to-end at N=1024 + shaping ON; partner action_dim 8 resolved; healthy learning signal; archive integral |
| P6 | Prereg tag untouched; `verify-prereg` blob-SHA check wired in the launcher | standing (ADR-007 §Discipline) |
| P7 | One process per cell (GPU PhysX once-per-process); chained launcher | standing (Rev 17; the campaign's chain pattern) |

## §4 — Watch items with pre-stated reads (founder addition 2)

- **W-A — training place_max decline at α\*=0.5** (0.83 unshaped → 0.55–0.62 shaped, while
  cold place held 30/30). **Pre-stated read: if any gate cell shows cold place < 27/30, that
  is the α-too-big signature firing at gate scale** — named here so it is recognised, not
  discovered mid-analysis. It would be reported per cell in the gate archive and would gate
  any follow-on α discussion; it does not retroactively alter the spike's prereg'd
  success-rate measurement.
- **W-B — drift-on-settle skill gradient.** The rider-1 joint-occupancy instrument
  (per-step placed/static/placed∧static; P(static | placed) and P(placed | static)) **is
  carried into every gate cell's logging** so the gap analysis can decompose conversion
  differences between conditions (e.g. a hetero−homo gap driven by settle-retention rather
  than grasp/place). Diagnostic only; the gate metric remains prereg'd IQM success.
- **W-C — IQM-on-near-binary deltas** can pin to 0 on borderline data (the `next-stage`
  module's own caveat); revisit only if the gate result is borderline.

## §5 — Budget and execution shape

10 cells (2 conditions × 5 seeds) × 20M frames ≈ 33 min/cell ≈ **5.5 h chained on the local
RTX 2080 SUPER** (one process per cell, halt-without-retry). **The A100 question
(REMEDIATION_LOG §8.4) is dissolved for AS.** Artifacts per cell: SpikeRun JSON via the
production `chamber-spike run --sub-stage 1b` path, per-window JSONL, rider-1 eval-step
JSONL, trimmed log; collision-proof run_ids; `SHA256SUMS.txt`; archive at
`spikes/results/stage1-AS-<launch-date>/` per the established layout.

## §6 — The measurement, and what each outcome triggers (surfaced; decided by no one here)

`chamber-spike next-stage --prior-stage 1` over the spike's SpikeRun archives:

- **Exit 0 (≥20 pp CI-low gap on AS):** Stage 2 unlocks per ADR-007 §Validation criteria —
  Stage-2 CR/CM prereg tag-cutting becomes available (P9 discipline; new tags, new ADR
  consultation per §Discipline). Surfaced, not decided.
- **Exit 5 (<20 pp):** **this run arms the R-RES-01 trigger for the first time** — the risk
  register's early-warning condition is a post-rig-validation firing closing <20 pp on both
  axes, and this would be the first gate-scale firing on a rig with every known defect fixed
  and consolidation demonstrated. The pre-committed responses are named, not chosen: the
  ADR-007 §4a runbook (immutable firing archive; consultation discipline — and note the §4a
  third-firing consultation requirement already bound once and was founder-waived for Rev 15;
  a gate-scale miss re-engages it), and the ADR-008 fallback ladder for the estimator
  (BootstrapCI carries mean alongside IQM; any estimator change is ADR-level). A sub-20 pp
  result with healthy per-cell success would be a *finding about the axis*, not the rig —
  exactly the distinction the register's trigger exists to mark.

## §7 — Stop discipline

This document authorises nothing. Launch requires: P1–P2 merged, founder approval of this
spec **explicitly and separately**, and the launcher invoking the prereg-verbatim protocol.
Between approval and launch the §2 regime is frozen; the only permitted change to this file
before launch is via a new approval-record file (the I8 guard-hook pattern).

## Cross-references

`spikes/preregistration/AS.yaml` @ `prereg-stage1-AS-2026-05-15`; ADR-007 §Validation
criteria + §Discipline + §Stage 1b Revs 14–18; ADR-008 §Decision (bootstrap/IQM; fallbacks);
ADR-016 (sub_stage); ADR-002 (provenance; GPU 95 %-CI caveat); R-RES-01 (planning-kit risk
register; referenced by ID — kit not mounted on this host); PRs #217 / #219 / #220;
`HOMO_SMOKE_2026-06-11.md`; the campaign chain:
`../2026-06-10-regime-alignment/` → `../2026-06-10-success-static-probe/` →
`../2026-06-11-gamma-scan/` → `../2026-06-11-pbrs-settle/`.
