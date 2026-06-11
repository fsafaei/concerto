# Stage-1b AS gate spike — verdict report

**Date.** 2026-06-11 (spike 13:43–18:45 UTC, 10/10 cells, single production process, zero
failures; W-A/W-B replays 18:51–19:29 UTC).
**Author.** Farhad Safaei.
**Status.** The gate-scale verdict record for the spike launched under
`GATE_REGIME_SPEC_2026-06-11.md` (founder-approved; PR #221) at the frozen regime (PR #222
launch yaml; launch SHA `460af5e`). **This document records the measurement and the
pre-stated watch-item reads. It decides nothing** — no Stage-2 action, no consultation
artifact, no register change is taken here (spec §6/§7; founder stop discipline).

## §1 — The measurement (prereg'd protocol, verbatim)

Protocol per `spikes/preregistration/AS.yaml` @ `prereg-stage1-AS-2026-05-15`
(blob SHA `29e397a4…` verified by `verify_git_tag` at launch): both condition_ids, seeds
[0,1,2,3,4], 20 episodes/seed/condition = 100 episodes per condition; IQM success; cluster
bootstrap; gate on the 95 % CI lower bound of the homo−hetero gap ≥ 20 pp.

**`chamber-spike next-stage --prior-stage 1` → exit 5 — the Stage-1 gate FAILS.**

```
chamber-spike next-stage: FAIL -- Stage-1 gate (gate_pp=20.0); 0/1 axis(es) pass.
  axis=AS  ci_low_pp=-2.00  ci_high_pp=0.00  (fail)
```

(`next_stage_measurement.txt`, this directory, verbatim.)

## §2 — Per-cell table

Gate columns are the prereg'd 20-episode instrument (the SpikeRun); replay columns are the
W-A/W-B diagnostic instrument (30 cold episodes from each cell's terminal checkpoint,
step 19,999,744, with rider-1 occupancy logging — sidecars in this directory).

| cell | gate success (20 ep) | replay success (30 ep) | replay cold grasp / place (30 ep) | P(static∧placed \| placed) | P(placed \| static) |
|---|---|---|---|---|---|
| HOMO s0 | 0/20 | 3/30 | 30/30 / 30/30 | 0.0012 | 1.00 |
| HOMO s1 | 0/20 | 0/30 | 30/30 / 30/30 | 0 | — |
| HOMO s2 | 0/20 | 0/30 | 30/30 / 30/30 | 0 | — |
| HOMO s3 | 1/20 | 1/30 | 30/30 / 30/30 | 0.0004 | 1.00 |
| HOMO s4 | 0/20 | 0/30 | 30/30 / 30/30 | 0 | — |
| HETERO s0 | 3/20 | 4/30 | 30/30 / 30/30 | 0.0020 | 0.44 |
| HETERO s1 | 7/20 | 11/30 | 30/30 / 30/30 | 0.0062 | 1.00 |
| HETERO s2 | 0/20 | 0/30 | 30/30 / 30/30 | 0 | — |
| HETERO s3 | 2/20 | 2/30 | 30/30 / 30/30 | 0.0012 | 0.29 |
| HETERO s4 | 0/20 | 0/30 | 30/30 / 30/30 | 0 | — |

**Pooled gate success: HOMO 1/100, HETERO 12/100.** The prereg'd gap (homo − hetero) is
**negative, ≈ −11 pp pooled** — the heterogeneous condition outperforms the homogeneous one.
The pre-stated **W-C** caveat fired exactly as written: IQM over near-binary per-pair deltas
pins the CI to [−2.00, 0.00] while the pooled mean gap is ≈ −11 pp (`BootstrapCI` carries the
mean alongside the IQM for any ADR-008-level follow-up; no estimator change is proposed or
taken here).

## §3 — Watch-item reads (pre-stated in the spec §4; answered)

- **W-A — does NOT fire.** Cold place is **30/30 on all ten cells, both conditions** (and
  cold grasp likewise). α\*=0.5 shows no α-too-big signature at gate scale; grasp and place
  are fully consolidated, condition-symmetrically. The entire homo−hetero difference lives in
  the settle-conversion step.
- **W-B — the decomposition the rider instrument was built for.** Hetero's converting seeds
  reproduce their PBRS-slice seed-level profiles under independent retraining (s0: 4/30 with
  P(placed | static) = 0.44; s1: 11/30 at 1.00 — the same per-seed outcomes as the
  2026-06-11 PBRS-settle arms at the same seeds and config). Homo's rare successes hold
  placement perfectly while static (P(placed | static) = 1.0 where defined) — homo's deficit
  is that it almost never *attempts* settling, not that settling costs it placement.
  **Conversion, not retention, separates the conditions.**

## §4 — Records of note (facts, not findings)

1. **Provenance stamp drift.** Per-cell `git_sha` stamps span four values (`460af5e`,
   `0fa53fa`, `b560466`, `191c614`) because `compute_run_metadata` executes
   `git rev-parse HEAD` at each cell's start, and local git activity (PR merges) moved HEAD
   during the 5-hour single-process run. The training code and config were loaded once at
   process start (Python module cache; the cfg object is resolved once in `run_axis`), so
   the cells are scientifically identical to the launch vintage; the launch SHA `460af5e`
   (`chain_timeline.log`) is authoritative. Engineering follow-up implied (not filed here,
   per the stop discipline): capture the SHA once per process.
2. **R-RES-01.** AS closed < 20 pp at gate scale on a rig with every known defect fixed and
   both conditions structurally healthy. The register's early-warning trigger reads
   "< 20 pp on **both** axes"; OM has never fired (blocked independently by #177). Whether
   the trigger formally arms is a founder register decision, recorded nowhere in this
   document.
3. **Sign reversal is the substantive observation.** The prereg'd hypothesis (homogeneity
   advantage ≥ 20 pp) is not merely unmet — the measured direction is reversed, with W-A
   confirming both conditions consolidate grasp+place perfectly. Per the spec §6, this is a
   finding about the **axis**, not the rig. Interpretation, framing, and next steps are
   deliberately absent from this report.

## §5 — Archive manifest (this directory; `SHA256SUMS.txt` covers all)

`spike_stage1_as_2026-06-11.json` (the SpikeRun; prereg SHA embedded; 200 episodes);
`next_stage_measurement.txt` (verbatim output + exit code); `train_{cond}_seed{N}_{run_id}.jsonl`
× 10 (per-cell training streams; fingerprinted run_ids, all distinct);
`wab_{cond}_seed{N}_results.json` + `wab_{cond}_seed{N}_eval_steps.jsonl` × 10 (the W-A/W-B
replay instrument); `launch_trimmed.log`; `chain_timeline.log`.

## Cross-references

`GATE_REGIME_SPEC_2026-06-11.md` (PR #221) §1/§2/§4/§6/§7; PR #222 (launch yaml);
`spikes/preregistration/AS.yaml` @ `prereg-stage1-AS-2026-05-15`; ADR-007 §Validation
criteria + §Discipline + §Stage 1b Revs 14–18; ADR-008 §Decision (IQM/bootstrap; mean carried
for follow-up); ADR-016 (sub_stage); ADR-002 (provenance; the stamp-drift record above);
the campaign chain `spikes/results/stage1-failure-investigation/2026-06-10-regime-alignment/`
→ `2026-06-10-success-static-probe/` → `2026-06-11-gamma-scan/` → `2026-06-11-pbrs-settle/`
→ `2026-06-11-gate-regime-spec/`.
