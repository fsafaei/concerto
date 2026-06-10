# Gate-spike readiness note — Stage-1b AS at the aligned regime

**Date.** 2026-06-10.
**Author.** Farhad Safaei.
**Status.** DRAFT — readiness assessment only; **the gate spike does not launch from this
document** (I1). Builds on the merged `FOLLOWON_NOTE_DRAFT_2026-06-10.md` §(b) and resolves its
two named blockers by reference to the probe (`SUCCESS_STATIC_PROBE_2026-06-10.md`) and the
options note (`OPTIONS_NOTE_C1_2026-06-10.md`).

## Checklist (verified, with evidence)

1. **Rev 17 merged.** ✅ PR #213 merged 2026-06-10T21:10Z (`e82d0ed`); ADR-007 §Revision
   history carries the entry; PR #212 (evidence) merged 21:01Z.
2. **Vectorised training reachable through the production `TrainedPolicyFactory` dispatch —
   exercised, not assumed.** ✅ with one new hard blocker found by exercising it
   (`factory_dispatch_verification.json`): `_compose_cfg_for_call` preserves
   `env.num_envs` per (seed, condition) cell; the factory's `__call__` trains at
   `num_envs=16` through the production `run_training`/`build_env` path and the returned
   deterministic closure drives a 1-env eval episode. **BUT (B1, hard blocker):** the spike
   adapter `_run_axis_with_factories` builds the per-cell **eval env before invoking the
   factory**, and SAPIEN GPU PhysX can only be enabled before any CPU-PhysX scene exists in
   the process — so at `num_envs > 1` the production adapter order is fatal
   (`GPU PhysX can only be enabled once before any other code involving PhysX`). The
   regime-alignment runs worked because their driver built the training env first. Fix
   options (small, one PR, ADR-007-noted as a Rev 17 implementation detail): (a) reorder the
   adapter to invoke the factory before building the eval env; (b) run the factory's training
   in a subprocess; (c) build the eval env on the GPU sim backend. Until one lands, the gate
   spike cannot dispatch through `chamber-spike run --sub-stage 1b`.
3. **Condition symmetry pinned.** Per Rev 17 the launcher must apply identical regime settings
   to both AS conditions: `num_envs=1024`, 20M frames, the chosen γ (see B3), 32-per-env
   rollouts (32,768-transition updates), and the **filter-off training posture** under the
   Rev 17 operator-override record — unless the founder first closes **D-034**
   (`action_linf_component`) AND a batched filter exists; neither holds today, so filter-off
   is the gate posture of record, identically on both conditions. **OM:** the same symmetry
   clause applies if OM fires in the same window, but OM remains blocked on the vision-head
   trainer (P1.05.6 / issue #177) — OM cannot fire in this window, and the gate rule needs
   only one axis (§6).
4. **The prereg'd protocol governs the gate.** The gate runs the verbatim
   `spikes/preregistration/AS.yaml` protocol: the two AS `condition_id` strings, the 5-seed
   list, 20 episodes/seed/condition (100 episodes per condition). The 30-episode cold
   instrument was the investigation's measurement tool, not the gate's; nothing from the
   probe/characterization replaces the prereg instrument.
5. **RUNID-COLLISION (B2, hard blocker).** `compute_run_metadata` derives `run_id` without
   hashing the config, so same-seed cells at the same commit collide — a gate archive with 10
   cells across 5 shared seeds would collide pervasively. The config-hash/nonce fix
   (ADR-002-adjacent, provenance not determinism) **lands before the gate spike**;
   arm-disambiguated filenames were the investigation's workaround, not a fix. Filed as issue #214.
6. **Budget arithmetic at measured throughput.** 0.5 GPU-h per 20M-frame cell on the local
   RTX 2080 SUPER (33 min measured, 11.1k steps/s sustained, ~2.7 GB device memory): the
   10-cell AS gate (5 seeds × 2 conditions) ≈ **5.5 h local, single GPU, chained**. **The
   A100 question (REMEDIATION_LOG §8.4) is dissolved for AS** — the existing 5 A100-h/axis
   budget is no longer the binding resource; the local card covers the axis in one working
   day with margin. OM adds nothing to this arithmetic until P1.05.6 ships a vision-head
   trainer (whose training cost is unknown and will need its own benchmark).
7. **What `chamber-spike next-stage --prior-stage 1` requires to exit 0.** It reads SpikeRun
   JSON archives, skips `sub_stage="1a"` runs, recomputes the paired-cluster bootstrap per
   axis on per-episode data, and exits 0 iff **at least one of {AS, OM} shows
   `ci_low_pp ≥ 20.0`** (IQM aggregate; exit 5 otherwise; all-1a inputs exit 5 with the
   ADR-016 message). So AS alone can clear Stage 2 — consistent with §3's OM deferral. Note
   the module's own caveat: IQM on near-binary per-pair deltas can pin to 0; with success
   rates off the floor this is a watch item, not a blocker.

## Blockers vs watch items

**Hard blockers (gate cannot launch until closed):**

- **B1 — adapter env-build order at `num_envs > 1`** (item 2). Small code fix + Tier-2 test;
  its own `fix` PR (issue #215, which also carries W1 as a rider).
- **B2 — RUNID-COLLISION** (item 5). Small code fix in `compute_run_metadata`; ADR-002
  §Revision-history entry; its own `fix` PR.
- **B3 — the success-conjunct floor (the scientific blocker; critical path).** The gate
  measures the homo−hetero **success** gap, and success is structurally ≈0 on both conditions
  at the regime that consolidates place (C1, probe verdict). Until the founder picks from the
  options note (γ scan / 40M extension / PBRS settle term) and the chosen path demonstrates
  non-floor success on at least the hetero cell, a success-gap gate measures noise. **The
  C1 decision precedes the gate-regime spec; the gate's γ is an output of that decision, not
  an input to be defaulted.**

**Watch items (not blockers):**

- **W1 — final-checkpoint bucket miss.** The vectorised loop's frame-bucketed checkpointer
  never fires the terminal bucket when `total_frames` is not reached exactly
  (19,531×1,024 = 19,999,744 < 20M ⇒ last checkpoint at 19.0M). The in-memory policy is what
  the factory evaluates, so gate results are unaffected — but archive reproducibility of the
  *final* policy wants a terminal checkpoint flush. One-line fix; can ride B1's PR.
- **W2 — IQM-on-binary-deltas pinning** (item 7) — revisit only if gate data is borderline.
- **W3 — GPU 95 %-CI determinism contract** (ADR-002/ADR-007) applies to all gate cells;
  unchanged, restated for the record.

## Sequencing implied (founder cut)

C1 decision (options note) → B1+W1 fix PR and B2 fix PR in parallel → regime spec pre-statement
for the gate (γ fixed by the C1 outcome; condition-symmetric; prereg instrument) → gate spike →
`chamber-spike next-stage --prior-stage 1`.

## Cross-references

ADR-007 §Stage 1b Rev 17 + §Validation criteria (≥20 pp); ADR-008 §Decision (bootstrap/IQM);
ADR-016 §Decision (sub_stage typing); ADR-002 (provenance; GPU CI caveat); ADR-004 + D-034
(kit queue Rev 1.4 — referenced, not re-staged); `../2026-06-10-regime-alignment/`
(characterization, follow-on note); issue #177 (OM vision head);
`SUCCESS_STATIC_PROBE_2026-06-10.md`; `OPTIONS_NOTE_C1_2026-06-10.md`.
