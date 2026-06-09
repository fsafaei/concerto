# Stage-1b AS — Grasp-Remediation Campaign Log

**Snapshot date.** 2026-06-09 (PA3 in progress; this is a point-in-time record for the
consultation — results that land after this date are added as NEW files in this directory per the
firing-archive immutability rule, not by editing this one).
**Author.** Farhad Safaei.
**Scope.** The empirical campaign *after* the env-truncation root-cause fix (PR #206) to make
Stage-1b AS-hetero grasping reliable and convert grasps into pick-place success. Follows the
2026-05-24-p1-05-9 firing and its `CONSULTATION_BRIEF_RESOLUTION_2026-06-06.md`.
**Discipline note.** The formal third-§4a senior-advisor consultation gate was waived for the
truncation slice by founder decision (ADR-007 Rev 15); the founder is now running the remediation
arms empirically. This log is the material for a consultation on the **remaining** problem
(grasp reliability + place conversion), which the truncation fix did *not* solve.

---

## §0 — TL;DR for the consultation

- **Root cause (fixed, landed).** The env never enforced `episode_length` (`BaseEnv.step` hard-codes
  `truncated=False`); training was a single-config non-terminating MDP with collapsed advantages.
  Fixed in PR #206 (ADR-007 Rev 15, ADR-002 amendment). The MDP is now properly episodic.
- **The fix is NECESSARY but NOT SUFFICIENT.** With the fix, grasping becomes *possible* but is
  **seed-fragile** (1 of 3 seeds grasps ~10%; the others 0), and full pick-place **success stays at
  the floor** (~0.2%, spawn-coincidence noise) on every seed.
- **Two remediation arms tried, both FAILED and REGRESSED the metrics:**
  - **PA1** (reach→grasp dense reward bridge): non-PBRS shaping drove the policy into a deceptive
    low-value optimum (advantages collapsed); still 0 grasp.
  - **PA2** (high-early→annealed entropy schedule): too much entropy *prevented* grasp commitment —
    it made the *grasping* seed stop grasping. Exploration-via-entropy refuted.
- **Current arm (PA3, IN PROGRESS):** a reset-state grasp **curriculum** — seed a scheduled,
  annealed fraction of episodes from a validated pre-grasp pose so closure is *sampled and
  reinforced* directly (reward/optimum untouched; anneals to cold). Early signal positive
  (`grasp_rate ~0.10` from the assisted starts). **Decisive read pending:** does the
  curriculum-trained policy grasp **cold** at eval?
- **Open question for the consultation (see §6/§8):** if PA3 also fails to produce *reliable cold
  grasping*, the evidence (neither reward-density, entropy, nor — pending — curriculum works) would
  point at a harder problem (demonstrations/BC, action-space change, or a re-scoping of the
  Stage-1b task). Place-stage conversion (PB1, PBRS) is gated behind reliable grasping and untried.

---

## §1 — Baseline: the truncation fix and its validation

**The fix (landed, PR #206; ADR-007 Rev 15).** `Stage1PickPlaceEnv.step()` now truncates at
`episode_length` (via BaseEnv's `elapsed_steps`); eval horizon sourced from `DEFAULT_EPISODE_LENGTH`.
This makes episodes bounded and re-randomised per episode (root cause `ENV-NO-TRUNCATION`).

**Validation — 2 seeds → 3 seeds, 1M frames, truncation only (no bridge):**

| seed | grasp_rate max (windows) | place/success max | value_mean | advantage_std | dist_entropy | eval ever_grasped | episodes |
|---|---|---|---|---|---|---|---|
| broken baseline | 0 / 488 | 0 | 14.7 (= r/(1−γ)) | 0.001 (collapsed) | — | 0/5 | **1** |
| 0 | **0.098 (57)** | 0.002 / 0.002 | 15.0 | 0.187 | 4.06 | **1/10** | 10091 |
| 1 | 0.000 (0) | 0.002 / 0.002 | 5.2 | 0.031 | 4.01 | 0/10 | 10090 |
| 2 | 0.002 (3 ≈ noise) | 0.002 / 0.002 | 14.1 | 0.152 | 4.01 | 0/10 | 10080 |

**Reading.** Pathology resolved on all seeds (episodic MDP, advantages alive, value is the bounded
return not the infinite-horizon runaway). But grasping is **seed-fragile** (1/3) and **place/success
never leaves the floor**. Mechanistic split (from the analyst): seed 0 drives `gripper_cmd_mean → −0.89`
(commits to closure, grasps from ~step 574k, still climbing at 1M); seeds 1/2 command the gripper
*open* and never commit. Entropy rises on all (not collapse).

**H3 horizon re-eval (refuted).** Loading seed-0's checkpoint and re-rolling at H=100/200/300: the
grasp happens at **step 1–2** (early), with 98–299 steps to spare, yet **never places at any
horizon**. So the grasp→place stall is **credit/exploration, not horizon** — extra time does not
help, de-prioritising any horizon change.

---

## §2 — Remediation arms tried (both failed)

### PA1 — reach→grasp dense reward bridge (additive, non-PBRS), w=0.15, seed 1

| seed 1 | grasp_rate max | value_mean | advantage_std | eval grasp |
|---|---|---|---|---|
| baseline (no bridge) | 0.000 | 5.17 | 0.031 | 0/10 |
| **PA1 (bridge w=0.15)** | **0.000** | **2.06** | **0.004 (collapsed)** | **0/10** |

**Verdict: FAILED + regressed.** The additive (non-potential-based) closure reward steered the
policy into a **deceptive low-value local optimum** (collect a little closure reward near the cube
without grasping), where advantages collapse toward the broken-MDP level and learning stops
("collapse-and-stop"). Demoted to contingency; if any closure shaping survives it must be PBRS.

### PA2 — high-early→annealed entropy schedule (`entropy_coef` 0.05 → 0.005), seed 0

| seed 0 | grasp_rate max | value_mean | advantage_std | dist_entropy | eval grasp |
|---|---|---|---|---|---|
| baseline (entropy 0.01) | **0.098 (57)** | 15.0 | 0.187 | 4.06 | 1/10 |
| **PA2 (anneal 0.05→0.005)** | **0.000 (0)** | **2.10** | **0.007** | **4.92** | **0/10** |

**Verdict: FAILED + regressed — on the *easy* (grasping) seed.** High early entropy kept the policy
too stochastic (final dist_entropy 4.92 vs 4.06); it **never committed** to closure, value collapsed,
advantages collapsed to 0.007 (the abort signature). **Exploration-via-entropy is refuted:** the
grasp is a fragile *commitment*; more entropy pushes the policy *away* from it. PA2 seed 1 was
skipped (the seed-0 regression was decisive; founder approved skipping).

**Synthesis of PA1 + PA2.** The grasp is a **fragile commitment** that *both* denser reward (PA1) and
more exploration (PA2) destabilise. The problem is neither reward-density- nor entropy-limited.

---

## §3 — Current arm: PA3 reset-state grasp curriculum (IN PROGRESS)

**Hypothesis.** Grasp emergence is a rare-event the policy keeps failing to *discover*; rather than
shape reward (PA1) or add entropy (PA2), **change the state distribution** so the closure action is
*sampled and self-reinforced* directly (Florensa et al. 2017 reverse curriculum).

**Implementation** (branch `exp/p1-pa3-grasp-curriculum`, env-var-gated, committed; not on main).
- A valid **pre-grasp pose** was harvested from seed-0's checkpoint at a real grasp (the arm qpos),
  *validated* offline: arm at that pose + fingers open + cube placed at the resulting TCP + command
  close → `is_grasped` at step 0. (Sidesteps IK/orientation entirely.)
- A scheduled fraction `p_start·(1 − frames/total)` of episodes start from that pose (with
  per-episode joint jitter for grasp-position diversity), cube at the TCP between the open fingers;
  the rest are cold. `p_start=0.5`, annealed to 0 over 1M frames so the **final policy is cold**.
- Reward and optimum unchanged; RNG via `derive_substream` (P6). **Eval is forced cold** (the
  process-wide curriculum env var is nulled on the eval env) so `ever_grasped` measures the real,
  transferable skill, not the assisted starts.

**Status.** Seed 0 running (1M); the curriculum fires (`grasp_rate ~0.10` from assisted starts in
the first window — vs seed 1's 0 over a full cold 1M). Seed 1 chained next. **The decisive read is
the COLD eval `ever_grasped` at the end**, and whether the **cold-portion grasp_rate rises as the
curriculum anneals to 0**. Results will land as a new file in this directory.

---

## §4 — The decision-input chain (who said what)

- **Experiment-analyst** (`VALIDATION_ANALYSIS_truncation_2seed`, .local): pathology resolved; the
  gap is a grasp-emergence reliability problem, seed-divergent at 1M; ranked H1 exploration
  bottleneck, H2 place-credit chain, H3 horizon (later refuted), H4 entropy.
- **Remediation-designer** (`REMEDIATION_PROPOSAL_2026-06-07`, .local): rev-1 ranked PA1
  (bridge)/PB1 (place PBRS); rev-2 after PA1 failed re-ranked to **PA3 curriculum rank-1, PA2
  entropy rank-2**, demoted the additive bridge to PA4/contingency, put **H4 exploration
  co-leading**. The bridge survives only as PBRS (PA4).

---

## §6 — Levers not yet tried (the remaining ladder)

1. **PA3 (in progress)** — reset-state curriculum (grasp reliability).
2. **PB1 — PBRS stage-potential place shaping** (grasp→place conversion). **Gated behind reliable
   grasping** — moot at `grasp_rate ≈ 0`. Untried.
3. **PA4 — PBRS closure term** (potential-based; cannot move the optimum like PA1 did). Contingency.
4. **Not designed / consultation-level options** if PA3 fails: learning-from-demonstration / BC
   seeding (we *have* validated grasp trajectories from seed 0); an **action-space change**
   (ee-delta control to shrink the exploration manifold — flagged by the designer, breaks AS-prereg
   comparability → new ADR/prereg tag); or **re-scoping** the Stage-1b task (e.g. reach-only or
   grasp-only gate) — the heaviest, ADR-007-level decision.

---

## §7 — References

- **Root-cause fix:** PR #206 (on `main`); ADR-007 §Revision history Rev 15; ADR-002 2026-06-06
  amendment; `2026-06-06-missing-gripper/` (embodiment cleared).
- **Firing record:** `../2026-05-24-p1-05-9/` (THIRD_FIRING.md + CONSULTATION_BRIEF_RESOLUTION).
- **Experiment branches:** `exp/p1-pa1-grasp-bridge`, `exp/p1-pa2-entropy-anneal`,
  `exp/p1-pa3-grasp-curriculum` (each carries the arm's code; not merged to main).
- **Decision-input docs (gitignored `.local/experiment_analyses/`, operator-side; rsynced to the
  planning kit):** `VALIDATION_ANALYSIS_truncation_2seed_2026-06-07.md`,
  `REMEDIATION_PROPOSAL_2026-06-07_truncation-validation.md`, per-seed results JSONs,
  `pa3_pregrasp_template.json`. These are local scratch — the committed numbers above are the record.

---

## §8 — Open questions for the consultation

1. **If PA3 fails to produce reliable cold grasping**, three levers in a row will have failed
   (reward-density, entropy, curriculum). Is the next step demonstrations/BC, an action-space change
   (new prereg tag), or a re-scoping of the Stage-1b success criterion? This is an ADR-007-level call.
2. **Place conversion (PB1) is entirely untried** because grasping isn't reliable. Even seed-0's
   ~10% grasp never places. Is a separate place-stage curriculum/PBRS warranted in parallel, or
   strictly after grasp reliability?
3. **What reliability bar clears the gate?** The AS gate needs a ≥20pp homo−hetero *success* gap;
   current hetero success is ~0. How much grasp/place reliability is "enough" to even measure a gap?
4. **Budget.** Each 1M seed is ~8 h on the local GPU; the gate spike is 5 seeds × 2 conditions on
   A100. At what point does the campaign move to A100 / a larger seed budget?
