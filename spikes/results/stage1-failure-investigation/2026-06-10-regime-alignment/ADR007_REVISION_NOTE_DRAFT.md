# ADR-007 §Stage 1b — draft revision-history entry (Rev 17): training-regime alignment

**Status.** DRAFT — not landed in `adr/ADR-007-*.md`; requires founder approval together with
`PRESTATEMENT.md`. Lands with the Phase-2 code PR (branch `feat/stage1b-vectorised-cell`,
commit `a1820ce5b47987e1aea75b8b038542c0f4175087`) once approved.
**Author.** Farhad Safaei.

---

## Proposed entry text (for ADR-007 §Revision history)

**Rev 17 (2026-06-10) — training-cell regime alignment (P1.05.10).** The Stage-1b
field-practice review (2026-06-10, planning kit) found the training cell sampling 10–50×
below every published baseline for this task class (1 env × 1M frames × γ=0.99 ×
1,024-transition updates, vs 1,024–4,096 envs × 10–50M × γ=0.8 × 16k–51k-transition updates
for ManiSkill PickCube-family tasks). This revision records:

1. **Training-cell parallelisation (`EnvConfig.num_envs`) and the per-task discount (γ) are
   implementation-details-of-the-cell** under the Rev 15 precedent (which ruled the same for
   the episode-horizon enforcement): they alter no prereg-locked surface. The prereg YAMLs,
   git tags, `condition_id` strings, seed list, and episode budget carry forward unchanged —
   **no tag rotation**. The comparison-protocol contract is extended with one mandatory
   clause: **both AS conditions train under identical regime settings** (same `num_envs`,
   frames, γ, rollout/batch geometry) in anything gate-facing — condition symmetry is what
   keeps the ≥20 pp gap comparison meaningful.
2. **Prior firings are not regime-comparable** with post-Rev-17 runs, exactly as Rev 15 ruled
   for the horizon: before/after tables must name the regime delta as the explanans, and the
   1-env fix-only triplet (Rev 15+16, 2026-06-10 characterization) is the designated
   single-variable baseline for it.
3. **Training-time safety posture at `num_envs > 1`.** The CBF-QP / conformal / braking stack
   is a per-pair single-env contract (ADR-004 §Decision snapshot map) and is not batched in
   this slice. Vectorised cells therefore run under the config's designed operator override
   `safety.enabled=false`; `concerto.training.ego_aht.train` loud-fails on
   `num_envs > 1 ∧ safety.enabled=true` (silent filter-dropping is forbidden). **For N>1
   training cells the audit-gate hook reads `safety_enabled=false` from the cell's JSONL
   summary and emits the non-failing "safety disabled by operator override; gate skipped"
   record per the existing kill-switch contract (Rev 7 / D10): the cell produces no
   training-time λ telemetry, predicates A and B are not evaluated, and the cell's gate
   evidence therefore carries no training-time safety-stack attestation — the operator
   override is itself the audit-trail record of that posture.** **Evaluation has always run
   unfiltered** — the deterministic eval closure
   (`chamber.benchmarks.stage1_common.TrainedPolicyFactory`, `trainer.act(obs,
   deterministic=True)` straight into the env) has never applied the training-time filter at
   any vintage (see `SAFETY_INTERFERENCE_PROBE_2026-06-10.md` §3.3) — **so eval-side gate
   evidence is unchanged by the N>1 training-time safety posture; deploy/eval-time safety
   architecture is untouched (ADR-004).** The probe also documents that the filtered training
   cell was operating under an uncalibrated `action_linf_component = 0.1` box clamp;
   recalibrating or removing that bound for any future filtered-training cell is an
   ADR-004-adjacent decision, explicitly out of scope here (queued in the planning kit's
   decision queue, 2026-06-10 entry).
4. **Determinism (ADR-002).** `num_envs = 1` cells are byte-identical to pre-Rev-17 builds
   (the vector paths are structurally unreachable). Vectorised cells draw per-episode
   cube/goal randomisation from per-env-index `derive_substream` streams, advanced only when
   that env resets — reproducible regardless of partial-reset interleaving; verified
   cross-process on GPU (`tests/integration/test_stage1_vectorised_real.py`). The GPU
   95 %-CI-overlap caveat (ADR-007 §Stage 1b) stands unchanged.
5. **Gate-spike implication.** The gate-scale spike spec (5 seeds × 2 conditions, A100, 5
   GPU-h/axis) adopts these regime settings condition-symmetrically. Measured local
   throughput (RTX 2080 SUPER: 11.1k steps/s at N=1,024, ~2.7 GB total device memory) implies
   the existing A100 budget covers field-scale sampling with ample margin (20M frames ≲ 0.5
   GPU-h/cell locally; A100 strictly faster).

## Why a revision entry and not a new ADR

Rev 15 established the category: a fix/alignment of the training rig that leaves every
preregistered comparison surface intact is an implementation detail of the cell, recorded in
§Revision history with the evidence trail, not a decision change. Parallelisation and discount
fall in the same category: they alter *how fast and how much* the cell samples, not *what is
compared*. ADR-007 stays Accepted.

## Evidence trail

`spikes/results/stage1-failure-investigation/2026-06-10-regime-alignment/PRESTATEMENT.md` (the
frozen experiment design this revision enables); `../2026-06-09-grasp-remediation/`
(RESETFIX_CHARACTERIZATION_seeds_2026-06-10.md, SAFETY_INTERFERENCE_PROBE_2026-06-10.md);
field-practice review (planning kit, 2026-06-10); PR #206 (Rev 15), PR #210 (Rev 16), PR #211
(fix-only characterization).
