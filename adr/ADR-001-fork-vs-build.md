# ADR-001: Fork ManiSkill2 vs. build standalone benchmark

**Status.** Proposed
**Authors.** Farhad Safaei
**Reviewers.** _to fill_
**Tags.** v0.2 §3.1, BENCH, infrastructure

## Context

Phase 0 — fork-vs-build is the gating infrastructure decision per v0.2 §3.1. Default position: extend ManiSkill2 if its abstractions admit our heterogeneity-axis controls (per-robot control rate, action space, observation modality, communication shaping) without monkey-patching internals; build standalone otherwise. Note: the lit-review cites "ManiSkill2" but the active codebase is now ManiSkill v3 (same repository, `haosulab/ManiSkill`, pinned to commit `a4a4f92`, version 3.0.1, 2026-04-20). This ADR targets v3 — the API changes are evolutionary, not breaking, for the extension strategy below.

## Considered alternatives

| # | Option | Source / advocate | Pros | Cons |
|---|--------|-------------------|------|------|
| A | Extend ManiSkill v3 | [Tier-2 #54], [ManiSkill audit] | Mature GPU-parallel rollout pipeline; multi-vendor robot asset zoo; multi-controller composition API; demonstrated multi-agent dispatch via `MultiAgent`; all four heterogeneity axes reachable via wrappers (~230 LOC total) without monkey-patching | `obs_mode` is env-global (not per-agent) — needs a texture-filter wrapper; per-robot control rate not native — needs `PerAgentActionRepeatWrapper`; ManiSkill2 lit-review citation is one major version stale |
| B | Build standalone on Isaac Lab | [Tier-3 #55, #56] | Full control over heterogeneity axes; modern PhysX 5 GPU backend | 3+ month infrastructure cost; forfeits robot asset library, GPU rollout pipeline, and demonstration tooling already built in ManiSkill |
| C | Fork BiGym | [Tier-2 #57] | Already bimanual-focused; VR teleoperation pipeline; five-category difficulty ladder directly usable for §3.2 | Single-embodiment (Unitree H1 only); no multi-robot or heterogeneous-embodiment support; lacks ManiSkill2's per-component controller API and SAPIEN+Warp-MPM stack |

## Decision

Extend ManiSkill v3 (the active successor to the lit-review's "ManiSkill2") by adding per-agent observation/action channel decomposition above the existing `MultiAgent` API. Three thin wrappers cover the four heterogeneity axes. Do not build standalone on Isaac Lab and do not fork BiGym as a primary base.

## Rationale

ManiSkill v3's `MultiAgent` API exposes a `spaces.Dict` action space keyed by agent uid and dispatches actions per uid with no shape coercion, meaning a partner zoo of heterogeneous embodiments (e.g., Panda + Fetch + Allegro) can be assembled by listing distinct `robot_uids` without touching framework code (`notes/tier2/54_maniskill2.md`). A dedicated technical audit (`maniskill_audit.md`, commit a4a4f92) confirmed all four B0 heterogeneity axes are reachable without monkey-patching: per-robot action space and per-robot sensor suite are native; per-robot texture-filter observation requires a ~30-LOC wrapper; per-robot control rate requires a ~50-LOC `PerAgentActionRepeatWrapper`; per-robot communication shaping requires a ~150-LOC additive wrapper over the obs dict. Building standalone on Isaac Lab would incur a 3+ month infrastructure penalty and forfeits GPU-vectorised rollouts across heterogeneous robot classes, the 20+ vendor robot asset library, and the demonstration tooling that ground B0's contact-rich tasks (stolen_ideas.md row 54). BiGym (paper 57) is a single-embodiment benchmark whose MuJoCo stack lacks the per-component controller API central to the §3.1 fork base, and it is retained only as a fallback if the Phase-0 smoke test fails (`notes/tier2/57_bigym.md`). The lit-review's contingent recommendation — "extend if abstractions admit heterogeneity-axis controls without monkey-patching" — is satisfied by ManiSkill v3 across all four axes.

## Evidence basis (links to reading notes)

- [notes/tier2/54_maniskill2.md] — multi-controller composition API; fork recommendation confirmed; primary §3.1 ADR seed; `stolen_ideas.md` row 54
- [notes/tier2/57_bigym.md] — BiGym is single-embodiment, lacks ManiSkill2 controller API; fallback only if spike fails; `stolen_ideas.md` row 57
- [notes/tier3/refs.bib #55 (RoboHive), #56 (RoboCasa)] — tier-3 cite-only; confirm Isaac Lab / standalone alternatives are lower-priority; no standalone notes
- [/Users/farhadsafaei/Desktop/CONCERTO/maniskill_audit.md] — axis-by-axis technical audit of ManiSkill v3 at commit a4a4f92; per-axis verdict and file-and-line index

## Consequences

- **Project scope.** B0 benchmark is built as an extension layer above ManiSkill v3. Team must write three wrappers (texture-filter obs ~30 LOC; `PerAgentActionRepeatWrapper` ~50 LOC; `CommShapingWrapper` ~150 LOC) before Phase-0 acceptance test. No physics layer rewrite.
- **v0.2 plan sections affected.** §3.1 is resolved by this ADR. §3.4 (heterogeneity-axis enumeration) and §3.9 (communication layer) receive their simulation scaffolding. The lit-review's "ManiSkill2" citations throughout §3 should be updated to reference ManiSkill v3 after the Phase-0 smoke test passes.
- **Other ADRs.** ADR-005 (simulator base) must select ManiSkill/SAPIEN as the physics backend; Isaac Lab is no longer a live alternative for the primary path. Revisit ADR-005 only if the smoke test fails.
- **Compute / time / hiring.** No 3-month infrastructure penalty from Option B. GPU vectorisation across heterogeneous `robot_uids` is confirmed available via ManiSkill's `build_separate=True` path. Phase-0 acceptance test is a ~50-line smoke script, estimated 1–2 days.

## Risks and mitigations

- [GPU vectorisation silently broken for heterogeneous `robot_uids` at partner-zoo scale ≥8 distinct uids] → mitigation: Phase-0 smoke test (see Validation criteria) instantiates a 3-robot env; extend to 8 uids before Phase-1 lock; fall back to BiGym or standalone if vectorisation fails at scale.
- [Global `_sim_steps_per_control` interacts badly with `PerAgentActionRepeatWrapper` at extreme rate ratios (10 Hz : 50 Hz mobile-base / hand)] → mitigation: include a controlled-friction handover micro-task in the smoke test at the 1:5 rate ratio; if wrapper path fails, use the `_load_agent` override path to supply per-robot `control_freq` directly (confirmed monkey-patch-free per `maniskill_audit.md` §3).
- [`build_separate=True` conflicts with dict-action dispatch in `_step_action` at partner-zoo randomisation time] → mitigation: test `build_separate=True` with `robot_uids` heterogeneous tuple in the smoke script; if conflict exists, isolate with a per-env wrapper that applies param randomisation before the action-dispatch call.

## Reversibility

Type-2 (two-way door). The ManiSkill v3 extension lives entirely in a wrapper layer (~230 LOC) above the physics engine; no modifications are made to `mani_skill/` internals. If the Phase-0 smoke test fails on any of the three verification items, the team can switch to BiGym (Option C) or Isaac Lab (Option B) within ~2 weeks by replacing only the env-adapter layer; RL algorithm code, partner-zoo infrastructure, and safety-filter code are environment-agnostic. The decision becomes Type-1 only after Phase-1 commences large-scale demonstration collection on the ManiSkill env — at that point, replacing the physics stack costs replay-buffer regeneration.

## Validation criteria

Phase-0 acceptance test (≤2 days): a 50-line script that (i) instantiates an env with `robot_uids=("panda_wristcam", "fetch", "allegro_hand_right")`; (ii) applies `PerAgentActionRepeatWrapper(rates={panda: 20, fetch: 10, hand: 50})` and `CommShapingWrapper(latency_ms=50, drop_rate=0.05)`; (iii) runs `env.step(stub_dict_action)` for 100 steps. Pass threshold — all three conditions hold with no runtime errors over 100 steps: (a) `obs["agent"]` contains three independently-namespaced proprio dicts; (b) `obs["comm"]` contains the shaped channel; (c) the slow agent's effective action update interval matches 1/rate to within one env tick. If any condition fails, retrigger the ADR toward BiGym or standalone.

## Open questions deferred to a later ADR

- ADR-005 must be updated to reflect Isaac Lab is no longer a live primary alternative; defer if multi-GPU rollout benchmarking between ManiSkill v3 and Isaac Lab is requested before Phase-1 lock.
- Whether `build_separate=True` at ≥8 heterogeneous `robot_uids` introduces silent rendering artefacts or physics drift — flagged as a smoke-test verification item above.
- Lit-review update to "ManiSkill v3" throughout §3 of v0.2 plan — human decision after smoke test passes; not an ADR concern.
