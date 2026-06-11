# What the AS-homo cell actually operationalizes — code facts

**Date.** 2026-06-11.
**Author.** Farhad Safaei.
**Status.** Phase-1 deliverable of the gate-verdict processing (§4a-runbook machinery; new
dated directory; I8). **Code facts with citations only — no interpretation beyond them**; the
two Phase-2 readings condition on this document.
**Question.** In the gate spike (`spikes/results/stage1-AS-2026-06-11/`), who learns in the
homo cell, and what — beyond partner embodiment — differs between the homo and hetero cells?

## §1 — Who learns: the ego alone, in BOTH conditions

The Stage-1b dispatch builds **one** `TrainedPolicyFactory` and uses it for every (seed,
condition) cell (`src/chamber/benchmarks/stage1_as.py:221-226`; the cells loop at
`stage1_as.py:333-334` calls the same factory per cell). The factory's per-cell override
(`stage1_common.py:530` `_compose_cfg_for_call`) rewrites only `seed`, `env.agent_uids`,
`env.condition_id`, and the partner's `uid`/`action_dim` — **`cfg.algo` is untouched and is
`"ego_aht_happo"` for every cell** (`concerto/training/config.py:454`). `run_training`
selects `EgoPPOTrainer.from_config` (`training_runner.py:296`) — a single HARL-HAPPO actor +
critic for the ego (`panda_wristcam`), with the ADR-009 frozen-partner gate as its first
construction step. The training-time partner in **both** conditions is the registry's
`scripted_heuristic` (`ScriptedHeuristicPartner`, deterministic, frozen, `target_xy (0,0)`);
the eval-time partner likewise (`stage1_as.py:346,558`).

**There is no MAPPO, no co-learning, and no parameter sharing between agents anywhere in the
homo cell.** The string `mappo_shared_param` exists only in the `condition_id` and in the
prereg YAML's free-form `notes`. The second panda never learns; it is scripted.

## §2 — Every difference between the cells beyond partner embodiment

1. **Partner *behaviour* differs, asymmetrically, via the known Cartesian-target mis-port.**
   The heuristic reads `obs["agent"][uid]["state"][:2]` as a planar xy
   (`heuristic.py:224`); under both Stage-1b embodiments that slice is the first two *joint
   positions* (the Rev 13 / ADR-002 2026-05-21 finding; ticketed as P1.05.10). Effect,
   measured on the production envs (probe in this session, zero-ego rollout):
   - **Hetero (fetch):** `state[:2]` ≈ (0, 0) at reset → zero action → **empirical no-op**
     (max |qvel| → 0 by step ~5). The Rev 13 finding, unchanged.
   - **Homo (panda_partner):** `state[:2]` = (0.0, **0.393**) — arm joints 1–2 of the Rev 16
     ready pose (`_PANDA_READY_QPOS[1] = π/8`). The heuristic emits a sustained
     `action[1] ≈ −0.39` driving **arm joint 2 toward 0**: the second panda's arm *rises
     from the ready pose toward upright and keeps moving through the shared workspace for
     the whole episode* (qpos[1] 0.393 → 0.116 by step 30, still decaying; partner max
     |qvel| 0.27 early, 0.09 at step 30 — never static within the horizon).
   So the homo ego trains and evaluates against a *perpetually moving* 7-DOF arm above the
   table, while the hetero ego faces a *motionless* base 0.62 m away. This is an
   **incidental implementation asymmetry**, not a prereg-pinned property.
2. **Observation dimensionality (inherent to embodiment, flows through identically-shaped
   machinery):** the AS synthesizer's widened ego state is 53-D homo (partner panda
   qpos 9 + qvel 9) vs 65-D hetero (fetch 15 + 15); actor/critic widths follow the obs dim;
   everything else about the obs pipeline (state_dict mode, keep-sets, no OM filter for
   either AS condition) is identical (`_CONDITION_TABLE`, `stage1_obs_filter.py`).
3. **Workspace geometry (inherent to embodiment):** the panda_partner's reach envelope
   covers the cube-spawn region; the fetch base never enters it. Base poses are mirrored at
   ±0.615 m for both (`_AGENT_BASE_POSE_XYZ`).
4. **Identical between cells:** algorithm and learner (ego-AHT HAPPO, factor=1), reward
   (panda-routed, `compute_normalized_dense_reward`), success predicate, regime (N=1024,
   γ=0.8, 20M, rollout/batch, PBRS α=0.5 — `config_hash`-verified per cell), safety posture
   (filter-off), reset contract (Rev 16 ready poses for both robots), episode horizon,
   eval protocol, partner *class* and target.

## §3 — Prereg-pinned vs incidental

- **Pinned by the locked YAML (`spikes/preregistration/AS.yaml` @
  `prereg-stage1-AS-2026-05-15`):** the two `condition_id` strings, the agent-tuple semantics
  they name (panda-only vs panda+fetch), seeds, episodes_per_seed, estimator
  (`iqm_success_rate`), cluster bootstrap, strict failure policy.
- **Described in the prereg's `notes` (part of the locked blob) but NOT implemented:** the
  notes specify the homo condition as "*shared-parameter MAPPO … the literature's
  homogeneous-zoo baseline*" — i.e. **two co-learning pandas** — while hetero is described
  as ego-AHT vs the scripted partner. The implementation runs **ego-AHT vs scripted partner
  in both conditions** (§1). The homo cell as executed is therefore *not* the baseline the
  prereg notes describe; it is "the hetero learning problem with a panda where the fetch
  stands" — plus difference §2.1.
- **Incidental implementation choices (no prereg basis):** the §2.1 partner-motion asymmetry
  (the P1.05.10 mis-port acting differently per embodiment); the single-factory/single-algo
  dispatch for both conditions (a P1.05 engineering decision, `stage1_as.py:213-227`).

## §4 — Empirical anchor

The §2.1 behaviour measurements come from a 30-step zero-ego rollout on the production envs
at this session's vintage (both conditions, `root_seed=0`); numbers quoted inline. The gate
cells' `config_hash` values confirm regime identity across all 10 cells (per-cell training
JSONLs, `spikes/results/stage1-AS-2026-06-11/`).

## Cross-references

`spikes/results/stage1-AS-2026-06-11/GATE_VERDICT_REPORT_2026-06-11.md`;
`spikes/preregistration/AS.yaml` @ `prereg-stage1-AS-2026-05-15` (schema fields + notes);
ADR-007 §Stage 1b Revs 9 (the 1b dispatch), 12 (widened state), 13 (the partner mis-port
finding) + ADR-002 §Revision history 2026-05-21 (the Tier-2 acceptance gap it formalised);
ADR-009 §Decision (frozen-partner contract); issue P1.05.10 (the mis-port ticket).
