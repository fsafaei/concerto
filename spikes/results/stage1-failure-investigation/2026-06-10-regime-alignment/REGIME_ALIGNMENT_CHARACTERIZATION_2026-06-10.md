# Regime-alignment characterization — fix-only at field-scale sampling

**Date.** 2026-06-10 (chain launched 16:27 UTC, completed 20:21 UTC; zero failures, zero
interventions, run order exactly per the frozen `PRESTATEMENT.md`).
**Author.** Farhad Safaei.
**Status.** New file in this directory (I8; `PRESTATEMENT.md` frozen at launch and untouched).
**One question (per the frozen pre-statement).** Does grasp consolidation appear once the
Stage-1b training regime is aligned with field practice — no curriculum, no reward change, no
entropy schedule, no demonstrations?

---

## §1 — Before/after: the 1-env fix-only triplet vs A1 (the regime delta only)

Single-variable baseline: the Rev 15+16 fix-only triplet
(`../2026-06-09-grasp-remediation/RESETFIX_CHARACTERIZATION_seeds_2026-06-10.md`). The A1 delta
vs that baseline is the regime bundle {1→1,024 envs, 1M→20M frames, γ 0.99→0.8,
1,024→32,768-transition updates, training-time filter off} — the §3 chain decomposes it.

| seed | fix-only 1-env: grasp max (nonzero win) / cold grasp | **A1: grasp max (nonzero win) / cold grasp** | **A1 cold placed** | A1 cold success | A1 place max (train) | adv_std / entropy / value |
|---|---|---|---|---|---|---|
| 0 | 0.061 (324/976) / 0/10 | **0.923 (608/610) / 30/30** | **30/30** | 0/30 | 0.835 | 0.089 / 3.89 / 2.69 |
| 1 | 0.040 (301/976) / 1/10 | **0.913 (609/610) / 30/30** | **30/30** | 0/30 | 0.827 | 0.086 / 3.68 / 2.60 |
| 2 | 0.034 (149/976) / 0/10 | **0.916 (608/610) / 30/30** | **30/30** | 0/30 | 0.832 | 0.106 / 3.60 / 2.62 |

Run ids: A1 seed 0 `96da3cb04741f9db`, seed 1 `796c17607a66261a`, seed 2 `d5ff0fee293503a6`
(per-window JSONLs archived under arm-disambiguated filenames — see §6). 20M frames per seed,
~33 min wall-clock each at N=1,024 (~11k steps/s sustained, RTX 2080 SUPER).

## §2 — Verdict, strictly per the frozen §5 rule

**The pre-stated bar:** ≥2/3 A1 seeds with cold `ever_grasped` ≥ 3/30.
**The result: 3/3 seeds at 30/30.** The **regime-explains-consolidation branch fires.**

Per the frozen rule, the grasp-remediation campaign **closes with zero levers**: no curriculum,
no reward bridge, no entropy schedule, no demonstrations were used or are needed for grasp
reliability. The next questions are place conversion at scale and the gate-spike regime spec —
both addressed by the follow-on note (`FOLLOWON_NOTE_DRAFT_2026-06-10.md`, founder cut).

Beyond the bar: cold **placement** also consolidated 3/3 at 30/30 (the staged reward's place
rung climbed on its own at γ=0.8 — training place_rate ~0.83). What did **not** consolidate on
any arm is the full `success` predicate (`is_obj_placed ∧ is_robot_static`): training
success_rate stays ≤0.0016 and cold success is 0/30 on all A1 seeds — the policy places but
never goes static. The bottleneck has moved from "can it grasp" through "can it place" to "does
it hold still after placing" — a far narrower question, recorded in the follow-on note, not
re-scoped here (I1).

## §3 — Diagnostic arms: the single-variable chain (cannot flip §2; recorded as evidence)

| cell | env count | frames | γ | training filter | cold grasp | cold placed | cold success | train grasp max / place max |
|---|---|---|---|---|---|---|---|---|
| fix-only seed 1 (baseline) | 1 | 1M | 0.99 | **on** (±0.1 box) | 1/10 | 0/10 | 0/10 | 0.040 / 0.003 |
| **A3** seed 1 | 1 | 1M | 0.99 | **off** | **30/30** | 1/30 | 1/30 | 0.620 / 0.098 |
| **A2** seed 0 | 1,024 | 20M | 0.99 | off | 30/30 | 16/30 | 2/30 | 0.930 / 0.098 |
| **A1** seeds 0/1/2 | 1,024 | 20M | **0.8** | off | 30/30 | **30/30** | 0/30 | ~0.92 / ~0.83 |

Reading each arrow (one variable class per step, per the founder's approved framing):

1. **baseline → A3 (the filter).** Removing the training-time safety stack — at the *old* 1-env,
   1M, γ=0.99 regime, same seed — takes cold grasping from 1/10 to **30/30** and training grasp
   from 0.040 max (301 nonzero windows) to 0.620 max (944/976 windows). This confirms the
   `SAFETY_INTERFERENCE_PROBE_2026-06-10.md` mechanism at full strength: the QP's uncalibrated
   ±0.1 action-box clamp (plus the train/eval dynamics mismatch) was the **dominant blocker of
   grasp consolidation**, independent of the sampling regime. Wall-clock corroboration: A3 ran
   1M frames in **101 min** vs ~8 h for the filtered baseline — the per-step QP solve was also
   ~5× of the old cell's wall-clock. (Diagnostic; feeds the queued `action_linf_component`
   decision, `DECISION_QUEUE_ENTRY_action_linf_2026-06-10.md`.)
2. **A3 → A2 (parallelism + budget).** At fixed γ=0.99 and filter-off, scaling 1→1,024 envs and
   1M→20M frames lifts training grasp 0.62→0.93 and cold placement 1/30→16/30. Sampling scale
   is what moves *place* progress at γ=0.99.
3. **A2 → A1 (the discount).** γ 0.99→0.8 at fixed scale completes place consolidation
   (16/30→30/30 cold; training place 0.098→0.83). The per-task discount is specifically a
   **place-consolidation** variable; grasping is already saturated on both sides of this arrow.
   (A2's PPO signature also shows the γ=0.99 pathology shrinking but present: value_mean 51.1
   (≈ horizon-inflated), entropy 5.08, adv_std 2.47 — vs A1's 2.6/3.7/0.09.)

## §4 — Standing confound note (founder decision 2026-06-10; no re-verdicting)

The PA1 and PA2 verdicts (`../2026-06-09-grasp-remediation/REMEDIATION_LOG.md` §2) now carry two
independent confounds — the broken reset-state init (fixed in Rev 16) and the ±0.1 action-box
clamp with unfiltered eval (probe + A3) — and remain suspended.

## §5 — PPO-health appendix (per arm, final-window values)

| run | advantage_std | dist_entropy | value_mean | n_episodes | wall-clock |
|---|---|---|---|---|---|
| A1 s0 | 0.089 | 3.89 | 2.69 | 202,013 | 33 min |
| A1 s1 | 0.086 | 3.68 | 2.60 | 202,312 | 33 min |
| A1 s2 | 0.106 | 3.60 | 2.62 | 202,272 | 33 min |
| A2 s0 | 2.466 | 5.08 | 51.11 | 201,024 | 32 min |
| A3 s1 | 2.580 | 3.90 | 20.93 | 10,024 | 101 min |

## §6 — Archived artifacts and one provenance finding

All files in this directory (I8; `SHA256SUMS.txt` covers everything):

- `{a1_seed0,a1_seed1,a1_seed2,a2_seed0,a3_seed1}_results.json` — per-run results/signature
  JSONs (30-episode cold-eval rows inline).
- `{arm}_{run_id}.jsonl` — per-window JSONLs, **arm-disambiguated filenames**.
- `launch_*_trimmed.log` (structural events + driver lines), `chain_timeline.log`.
- `PRESTATEMENT.md` (frozen), `ADR007_REVISION_NOTE_DRAFT.md`,
  `DECISION_QUEUE_ENTRY_action_linf_2026-06-10.md`, `FOLLOWON_NOTE_DRAFT_2026-06-10.md`.

**Provenance finding (RUNID-COLLISION).** `compute_run_metadata` derives `run_id` from
(seed, git_sha, …) but **not from the config**, so same-seed runs at the same commit collide:
A2 seed 0 reproduced A1 seed 0's `96da3cb04741f9db` and A3 seed 1 reproduced A1 seed 1's
`796c17607a66261a`. The driver's plain `{run_id}.jsonl` archive copies therefore overwrote; the
authoritative streams were recovered from each run's isolated workdir and re-archived under the
arm-disambiguated names above (contents verified distinct). No data was lost; results JSONs
were never ambiguous (arm-tagged filenames). **Follow-up:** hash the config (or accept a nonce)
into `run_id` — filed as an engineering follow-up alongside the code PR (ADR-002-adjacent;
provenance, not determinism).

## Cross-references

PR #206 (Rev 15) / #210 (Rev 16) / #211 (fix-only characterization); ADR-007 §Stage 1b
Revs 14–16 + `ADR007_REVISION_NOTE_DRAFT.md` (Rev 17, approved for the code PR); ADR-002
(determinism; GPU 95 %-CI caveat — A3's cross-run comparison is behavioural, not byte-level);
ADR-004 (deploy/eval safety posture untouched); `STAGE1B_FIELD_PRACTICE_REVIEW_2026-06-10.md`
(planning kit); `../2026-06-09-grasp-remediation/` (REMEDIATION_LOG,
RESETFIX_CHARACTERIZATION_seeds_2026-06-10, SAFETY_INTERFERENCE_PROBE_2026-06-10).
