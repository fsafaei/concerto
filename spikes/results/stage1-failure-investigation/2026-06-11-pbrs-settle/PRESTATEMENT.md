# PBRS-settle pre-statement — the pre-registered ladder lever, under NHR invariance

**Date.** 2026-06-11.
**Author.** Farhad Safaei.
**Status.** DRAFT — awaiting founder sign-off together with `ADR_NOTE_DRAFT.md`. Frozen
verbatim at launch (approval recorded as a new file per the approval-record pattern; I8 from
launch onward). **Nothing launches without explicit founder approval of BOTH documents.**
**One question.** Does a potential-based (policy-invariant) settle term convert the
one-action-away stillness (`SETTLE_REACHABILITY_PROBE_2026-06-11.md`) into reliable cold
success at the place-friendly γ=0.8 — closing the C1 incentive gap the γ-scan proved the
discount cannot close?
**Scope guard (I1).** This is the pre-registered ladder firing (REMEDIATION_LOG §6's
PBRS-class lever) under pre-statement + ADR discipline. Exactly two pre-stated α arms; no
other levers; no edits to the success predicate, static threshold, goal_thresh,
episode_length, partner, prereg YAMLs, or `SCHEMA_VERSION` (P10). **The canonical env reward
and `evaluate()` are byte-untouched** — the shaping is a training-cell-side, config-gated
transform (default off).

## Grounding (the complete chain)

Rev 15 (horizon) + Rev 16 (reset) fixed the rig; the clamp confound was isolated (A3); the
regime aligned to field practice (grasp+place consolidate 30/30; campaign closed with zero
levers); the success-static probe returned C1-pure (the `1−tanh(5·qvel)` valley); the γ-scan
returned **no gate-viable γ** (place collapses at γ≥0.95 before stillness pays; 2× budget
deepens the hold); the settle probe confirmed the threshold is **one zero-delta step away**
(floor 0.003 rad/s). The remaining gap is purely *temporal credit*: stopping pays ≈ the same
discounted total as holding fast (telescoping), but the payment is diffuse. PBRS re-profiles
exactly that — and, by Ng–Harada–Russell (1999, Thm 1), provably cannot move the optimum,
which is what PA1's non-potential bridge did (REMEDIATION_LOG §2).

## Design (frozen at sign-off)

**Potential.** Φ(s) = −α · min(‖qvel_arm‖_max, cap) · 1[is_obj_placed(s)], with
‖qvel_arm‖_max the max |qvel| over the ego's 7 arm joints (the `is_static` predicate's own
reduction; fingers excluded) and **cap = 0.7 rad/s** (covers the measured 0.40–0.66 hold
band with margin). Both factors are functions of the state only — the placement gate is a
state predicate — so Φ is state-only and NHR applies.

**Shaping.** F(s, s′) = γ·Φ(s′) − Φ(s) with **γ = 0.8, the training MDP's discount**
(invariance requires the MDP's γ; a mismatched shaping-γ forfeits the theorem). Added to the
per-step training reward in the training cell only.

**Terminal/truncation convention (explicit, the bug class this project keeps catching):**
Φ ≡ 0 at true termination — the terminal transition's shaping is F = −Φ(s_T) (γ·0 − Φ). At
**time-limit truncation** the episode does not terminate: Φ(s′) is evaluated on the actual
final observation (the auto-reset wrapper's `info["final_observation"]`, NOT the next
episode's reset obs), and the trainer's existing Pardo-style bootstrap then operates on the
shaped-reward MDP unchanged. No Φ-zeroing at truncation.

**Placement of the code.** A config-gated training-cell reward transform (trainer/wrapper
side; flag `shaping.settle_alpha`, default 0 = off → byte-identical pre-existing behaviour).
`Stage1PickPlaceEnv.compute_normalized_dense_reward` and `evaluate()` untouched; cold eval
measures the unshaped success predicate as always. Condition-symmetric by construction: the
same transform, same α, for AS-homo and AS-hetero in anything gate-facing (Rev 17 clause).

**α derivation (from the scan margins + settle constants; arithmetic shown).** Reward scale
after the env's /5 normalisation: placed-hold income ≈ 0.57/step; the static stage term pays
up to 0.2/step once still; the success jump is 1.0. The settle probe gives qvel_hold ≈
0.4–0.62 and a one-step settle to floor ≈ 0.003, so the **stop transition collects a one-time
bonus F_stop = α·(qvel_hold − γ·floor) ≈ 0.5α**, while sustained fast-hold collects
(1−γ)·α·qvel ≈ 0.12α/step — telescoping-equal in discounted sum, but the stop bonus is
front-loaded onto the deceleration action (the mechanism; see the ADR note).

- **α_lo = 0.1:** F_stop ≈ 0.05 (25 % of the per-step static term); max |Φ| = 0.07 (7 % of
  the success jump). The minimal-interference probe of the mechanism.
- **α_hi = 0.5:** F_stop ≈ 0.25 (one stage-term's per-step scale); max |Φ| = 0.35 — still
  well under the success jump (1.0) and under the discounted place income it could compete
  with. The assertive value if α_lo under-drives.

**Gated-potential transients (computed; founder may amend to the ungated variant):** entering
placement at speed 0.6 incurs a one-time F = −γ·α·0.6 ≈ −0.48α (α_hi: −0.24) — dominated ≥3×
by the discounted place income it unlocks (≈ 0.85) plus the success jump; un-placing at speed
collects +0.6α once but forfeits the same income. The ungated alternative
(Φ = −α·min(‖qvel‖, cap) everywhere) has no boundary transients but taxes the reach phase;
the gated form is the pre-stated primary because C1 is a *hold-phase* incentive defect.

**α-too-big signature (watched explicitly):** cold place < 27/30 on any α arm — the policy
decelerating instead of transporting. The instrument reports it per arm.

## Arms (frozen at sign-off)

All: γ=0.8, 20M frames, N=1024, fix-only + shaping, training filter off (Rev 17 posture),
AS-hetero, `rollout_length=32` / `batch_size=4096` / hidden 256 — the A1 regime exactly, plus
the one new flag.

| arm | α | seeds | budget |
|---|---|---|---|
| **C-lo** | 0.1 | 0/1/2 | 3 × ~33 min |
| **C-hi** | 0.5 | 0/1/2 | 3 × ~33 min |

Anchors (no new runs): A1 (α=0, γ=0.8, 20M ×3) and the γ-scan table. Run order, chained,
halt-without-retry, no mid-run changes: C-lo s0 → s1 → s2 → C-hi s0 → s1 → s2. ≈ **3.3 h**.

## Instrument (per run; identical to the γ-scan instrument)

Cold deterministic eval, 30 episodes/seed (prereg'd Rev 16 init, forced cold, `num_envs=1`,
unfiltered closure, **unshaped** success predicate) with per-step logging: cold
grasp/place/success ladder; P(static∧placed | placed); hold-qvel distribution vs the
0.2–0.7 band; training static_rate trajectory; PPO health; **the place-regression check**
named above. Artifacts as new files (I8): collision-proof run_ids (#214 fix), results JSONs,
per-window JSONLs, per-step eval JSONLs, trimmed logs, `SHA256SUMS.txt`.

## Decision rule (draft; FROZEN at launch)

- **Lever works:** some α with **≥2/3 seeds at cold place ≥ 27/30 AND cold success ≥ 3/30**.
  Tie-break: higher pooled success, then higher pooled place. → `GATE_REGIME_SPEC` drafted at
  (γ=0.8, PBRS-α\*) per Phase 4a; the gate spike launches only on separate founder approval.
- **Lever fails:** neither α qualifies. → The **senior-advisor consultation brief** is drafted
  (Phase 4b) — the evidence chain would then be complete: two rig defects fixed, the clamp
  isolated, the regime field-aligned, γ scanned, one policy-invariant lever tried at two
  calibrated α. **No further lever launches from this slice.**

## Budget and constraints

≈ 3.3 h chained on the local RTX 2080 SUPER, one process per run. I1/I6/I7/I8 as always; the
shaping code lands by its own `feat` PR with the ADR note (separate from evidence, per the
established pattern); `make verify-no-ai-mentions` green; signed Conventional Commits. Stop
points: founder sign-off before launch; verdict reported before Phase 4 drafting; nothing
gate-launching from this slice.

## Cross-references

`SETTLE_REACHABILITY_PROBE_2026-06-11.md` (this directory); `ADR_NOTE_DRAFT.md` (this
directory — approval co-requisite); `../2026-06-11-gamma-scan/` (no-γ\* verdict + margins);
`../2026-06-10-success-static-probe/` (C1; options note option C);
`../2026-06-10-regime-alignment/` (A1 anchors); ADR-007 §Stage 1b Rev 17;
REMEDIATION_LOG §2 (PA1) + §6 (the ladder entry firing here); Ng, Harada & Russell 1999.
