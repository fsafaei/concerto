# ADR-005: Simulator base — Isaac Lab, MuJoCo, PyBullet, ManiSkill

**Status.** Accepted (2026-05-13)
**Authors.** Farhad Safaei
**Reviewers.** _to fill_
**Tags.** v0.2 §3.1, §3.9, BENCH

## Context

Coupled to ADR-001 (fork-vs-build). ADR-001 (Proposed 2026-05-07) selected ManiSkill v3 as the benchmark fork base; that decision inherits SAPIEN 3 + Warp-MPM as the physics substrate. If ADR-001 had selected standalone build, simulator choice would remain open. This ADR documents the selection as a corollary and records the ruled-out alternatives for traceability. Note: the stub listed "MuJoCo (via Isaac Lab)" as an option — this conflates two distinct stacks. Isaac Lab runs on Isaac Sim / PhysX 5 (NVIDIA), not MuJoCo. The table below corrects this.

## Considered alternatives

| # | Option | Source / advocate | Pros | Cons |
|---|--------|-------------------|------|------|
| A | SAPIEN 3 + Warp-MPM (via ManiSkill v3) | [Tier-2 #54], [ADR-001] | Bundled with the ADR-001 fork choice; ~2500 FPS GPU-parallel rollout; rigid + soft-body; per-component controller API; 20+ vendor robot assets; `build_separate=True` for per-env randomisation | Coupled to ManiSkill v3 release cadence; `obs_mode` is env-global (needs wrapper) |
| B | PhysX 5 / Isaac Sim (via Isaac Lab) | [external — not in reading corpus] | Modern GPU-parallel PhysX 5; Isaac Lab ecosystem; full heterogeneity control | Standalone build cost ruled out by ADR-001 (3+ months); NVIDIA-hardware dependency; no manipulation robot asset library matching ManiSkill scale |
| C | MuJoCo (via BiGym or standalone) | [Tier-2 #57] | Industry standard for contact-rich control; MuJoCo 3.x supports GPU batching | Lower fidelity for contact-rich manipulation vs. SAPIEN PhysX 5; BiGym is single-embodiment and lacks ManiSkill2 per-component controller API; standalone MuJoCo requires physics-layer rebuild |
| D | PyBullet | [external] | Mature, simple API | No longer actively maintained upstream; no GPU vectorisation; slower than all other options for batch rollouts |

## Decision

SAPIEN 3 + Warp-MPM, as bundled with ManiSkill v3. This is a direct corollary of ADR-001's Decision to extend ManiSkill v3. No additional simulator integration work is required; the physics layer is inherited.

## Rationale

ADR-001 (Proposed 2026-05-07) established that ManiSkill v3's abstractions admit all four B0 heterogeneity axes without monkey-patching, making it the fork base of choice. SAPIEN 3 (rigid-body, PhysX 5 backend) + Warp-MPM (soft-body) is ManiSkill v3's physics substrate, delivering ~2500 FPS GPU-accelerated rollout on a single GPU — the throughput B0's whole-population evaluation requires (notes/tier2/54_maniskill2.md; stolen_ideas.md row 54: "fork rather than rebuild the SAPIEN physics + Warp-MPM stack"). Isaac Lab / PhysX 5 would match GPU throughput but requires the standalone build that ADR-001 ruled out at 3+ month infrastructure cost; no reading corpus note advocates for it as a primary path. MuJoCo via BiGym is the fallback simulator if the ADR-001 Phase-0 smoke test fails, but BiGym's single-embodiment constraint and absence of a per-component controller API make it a secondary choice, not a parallel primary (notes/tier2/57_bigym.md). PyBullet lacks GPU vectorisation and active upstream maintenance, disqualifying it for B0 batch evaluation. SAPIEN's `build_separate=True` path supports per-env physical randomisation across heterogeneous robot classes, which is required for B0's partner-zoo sweep (stolen_ideas.md row 54).

## Evidence basis (links to reading notes)

- [notes/tier2/54_maniskill2.md] — SAPIEN+Warp-MPM stack confirmed; ~2500 FPS; per-component controller API; fork recommendation; stolen_ideas.md row 54
- [notes/tier2/57_bigym.md] — MuJoCo (via BiGym) is fallback-only; lacks per-component controller API; stolen_ideas.md row 57
- [ADR-001-fork-vs-build.md] — upstream Decision that makes this ADR a corollary; ManiSkill audit (commit a4a4f92) as technical basis

## Consequences

- **Project scope.** No additional simulator integration required. Team writes wrapper layers above SAPIEN (total ~230 LOC per ADR-001) but does not touch the physics layer. Soft-body tasks (Warp-MPM) are available if §3.2 task set requires them; defer the decision on whether to activate them to ADR-015.
- **v0.2 plan sections affected.** §3.1 resolved by ADR-001; this ADR closes the simulator-choice sub-question. §3.9 (communication interface) operates above the SAPIEN obs-dict layer via the `CommShapingWrapper` — not a physics-layer concern.
- **Other ADRs.** Downstream of ADR-001; corollary of its Decision. ADR-003 (communication interface) and ADR-009 (partner zoo) work above the SAPIEN obs-dict and are unblocked. If ADR-001's Phase-0 smoke test fails and the fallback is BiGym/MuJoCo, this ADR must be retriggered simultaneously.
- **Compute / time / hiring.** Zero additional cost; SAPIEN is included in the ManiSkill v3 install. GPU-parallel rollout is available from day one.

## Risks and mitigations

- [SAPIEN 3 PhysX 5 contact dynamics differ from MuJoCo in dexterous-hand or high-friction regimes, causing policy transfer gaps if baselines are benchmarked elsewhere] → mitigation: for any §3.2 task that overlaps with a MuJoCo benchmark (e.g., BiGym tasks), record SAPIEN vs. MuJoCo success-rate delta in the task note; flag discrepancy if Δ > 10 pp.
- [ManiSkill v3 release cadence introduces breaking API changes between Phase 0 and Phase 1, breaking the ~230 LOC wrapper layer] → mitigation: pin to commit a4a4f92 for all Phase-0 work; upgrade only at Phase-0→1 gate review; wrapper layer is small enough (~230 LOC) to port within one day.

## Reversibility

Type-2 (two-way door). The wrapper layer above SAPIEN is env-agnostic — switching to Isaac Lab or MuJoCo requires replacing only the env adapter, not RL code, safety-filter code, or partner-zoo infrastructure. Becomes Type-1 only after large-scale Phase-1 demonstration collection on the SAPIEN env (same trigger as ADR-001), because regenerating the replay buffer would cost significant compute.

## Validation criteria

Resolved as a corollary of ADR-001's Phase-0 acceptance test: the 50-line smoke script instantiating a 3-robot heterogeneous env under SAPIEN 3 and running 100 steps without error. If the smoke test passes, SAPIEN is confirmed. No additional validation step is required beyond ADR-001. Failure of the smoke test triggers simultaneous retrigger of ADR-001 and this ADR.

## Open questions deferred to a later ADR

- Whether Warp-MPM soft-body support is needed for any §3.2 tasks — if not, the Warp dependency can be dropped at install time. Human decision at ADR-015 (tier-task scope freeze).
- SAPIEN vs. MuJoCo contact-calibration delta for dexterous-hand tasks — defer to §3.2 task implementation; not an ADR-level concern unless the delta exceeds 10 pp at the Phase-1 gate.

## Revision history

- 2026-05-13 status re-classification: status changed from Proposed to **Provisional** under the new ADR status taxonomy (see [ADR-INDEX §Status taxonomy](ADR-INDEX.md#status-taxonomy)); no Decision content is altered.
- 2026-05-13 lock: promoted to **Accepted (2026-05-13)** under the solo-developer working policy in ADR-INDEX (corollary of ADR-001; SAPIEN inherited via the merged ManiSkill v3 install); no Decision content is altered.
