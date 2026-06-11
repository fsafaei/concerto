# Addendum — PA1/PA2/PA3 ran under a non-canonical reset-state confound

**Date.** 2026-06-09.
**Author.** Farhad Safaei.
**Status.** Addendum to REMEDIATION_LOG.md (this directory). New file per the snapshot-immutability
rule in that log's header and I8; REMEDIATION_LOG.md and all prior firing directories are untouched.
**Relation to code.** Records the consequence, for this campaign's results, of the reset-state
initialisation bug fixed in PR #210 (ADR-007 §Stage-1b Rev 16; ADR-002 2026-06-09 determinism amendment).

## What was wrong during every run in this log

ManiSkill's `TableSceneBuilder.initialize()` dispatches robot init on `robot_uids` via a pure
if/elif chain with no terminal `else`, and carries no branch for the AS tuples — hetero
`("panda_wristcam","fetch")` or homo `("panda_wristcam","panda_partner")`. The Stage-1b env never set
robot qpos itself. So in **every** run recorded in REMEDIATION_LOG.md — the truncation-fix validation
(§1), PA1 (§2), PA2 (§2), and PA3 (§3) — each episode began with the ego Panda folded at the zero
configuration and the gripper **closed** (finger width 0.0; directly measured in
`../2026-06-06-missing-gripper/probe_embodiment.log`, E3). The canonical PickCube start — ready pose
with the gripper **open** — was never applied. This is a feasibility defect upstream of the reward,
exploration, and curriculum levers the arms varied.

## What it confounds

The arms' conclusions about their *levers* were drawn from a broken initial-state distribution and
should be suspended, not relied on:

- **Baseline (§1).** The "grasping is seed-fragile (1/3), place at the floor" characterisation that
  motivated the campaign was itself measured under the broken init. The one seed that grasped (seed 0)
  did so *despite* a folded-arm, closed-gripper start; seeds 1–2 never did. The fragility may be a
  symptom of the start state, not an intrinsic property of the task under a correct setup.
- **PA1 — reward bridge (FAILED + regressed).** The "deceptive low-value optimum / advantage collapse"
  was produced by adding a closure-shaping term to a policy that could neither reach (folded arm) nor
  close on the cube (gripper shut). The collapse-and-stop signature is consistent with a starved start,
  not only with the additive bridge being the wrong instrument.
- **PA2 — entropy schedule (FAILED + regressed).** "More entropy prevented grasp commitment" was read
  on a policy with no graspable start. The seed-0 regression is confounded.
- **PA3 — reset-state curriculum (early-positive).** Its positive signal is now explained: it seeds
  episodes from a validated pre-grasp pose with the fingers open, i.e. it **bypasses the broken reset**.
  PA3 was partly compensating for the init bug; its cold-eval read must be re-interpreted against the
  corrected init.

## What this addendum does not claim

It does not claim the init was the *only* obstacle (the single-env / 1M-frame sampling regime remains
far thinner than the published PickCube baseline of hundreds-to-thousands of parallel envs at ~10M
steps). It does not retroactively re-rank or re-verdict the arms — their archived records stand. It
does not authorise any new arm.

## The decisive test (pre-committed)

Per REMEDIATION_LOG.md §0/§C, the post-#210 single-seed cold-eval — fix only, no
bridge/entropy/curriculum, run on a seed that previously produced zero grasps (seed 1 or 2, not the
lucky seed 0) — determines whether grasping emerges from a correct start. If it grasps cold,
PA1/PA2/PA3 are moot for grasp emergence and the gate-scale 1M × ≥3-seed spike is warranted. If it does
not, the relevant arm(s) should be re-run on the corrected init before any lever conclusion is
recorded, and the §6/§8 consultation ladder (demonstrations/BC seeding, then an action-space change)
becomes the live question.

## References

- Fix: PR #210; ADR-007 §Stage-1b Rev 16 (reset-state contract; implementation-detail-of-the-cell ⇒
  no prereg tag rotation; both AS conditions affected ⇒ pre-fix gaps are not heterogeneity evidence);
  ADR-002 2026-06-09 determinism amendment.
- Defect evidence: `../2026-06-06-missing-gripper/MISSING_GRIPPER_INVESTIGATION.md` (E3, finger width
  0.0 at reset); ManiSkill `TableSceneBuilder.initialize()` dispatch gap.
- Campaign record: `REMEDIATION_LOG.md` (this directory), §1–§3.
