# Co-carry Rung-2 reward remediation — decision record (2026-06-16)

**Status:** implemented; gated by the GPU smoke + the train-to-reference stop criterion.
**Scope:** ADR-026 §Decision 4 (Phase-2 forward design); R-2026-06-B §15 Rung 2.
**Governance:** I1 (no Stage-1/M10), I3 (partner frozen during training), I8 (no
edits to existing `spikes/results/**` archives), I9 (no schema bump).

## Confirmed root cause (from the Rung-2 train-to-reference STOP)

`spikes/results/cocarry/rung2/cocarry_rung2_train_stop.json` (PR #246, commit
`e74501b`): at the full 300k budget the learned ego shows a **significant
positive learning slope** (+0.0119, p=1.0e-11, n=953) yet reaches **0% joint
success**. The eval shows *how* it fails:

- bar tilt p50 **13.4°** (the ego partly learns to brace the bar level),
- centroid-to-goal p50 **0.373 m** (vs the 0.10 m goal — **never transported**),
- wrist constraint-solver stress p90 **854 N** (**~6.5× the f_max 130 N
  ceiling** — the ego **fights** the frozen partner through the rigid weld).

The dense co-carry reward rewarded transport + level + settle + a success
bonus, and the success predicate gates on "unstressed" (`max_stress < f_max`),
**but the reward had no internal-stress term** — so the learner got **no
gradient against the fight**. Reward-up / success-flat: the policy climbs the
shaped reward (the partner does part of the transport + the ego's partial
motion) without ever satisfying the joint success conjunction. This is the
construct-validity signature the rung exists to surface.

## The fix (two principled, partner-agnostic additions)

Both penalise / shape a **physical quantity** (internal force, distance-to-goal),
never partner identity / policy / embodiment (spec §11 anti-leakage). Both are
named module constants in `chamber.envs.cocarry`, captured by the freeze
enumerator. **Coefficients set on a principled basis, NOT tuned to a target.**

### 1. Excess-internal-stress penalty (keystone) — `compute_normalized_dense_reward`
`penalty = COCARRY_REWARD_STRESS_COEFF · tanh(relu(stress − threshold)/scale)`,
subtracted from the reward sum (settle-window-gated, like `evaluate()`, so the
startup placement ring is not punished).

- `COCARRY_REWARD_STRESS_SOFT_THRESHOLD_N = COCARRY_STRESS_MAX_PROXY_N` (= f_max,
  130 N): the penalty bites exactly the stress that **violates the success
  constraint**. The committed Step-1 matched cooperative band (success-stress
  p50=92, p90=101, **p99=104.5**, max=105 N) sits ≥25 N below the threshold, so
  the penalty is **≈0 on the matched pair** — the 100% reference is unchanged —
  and bites only the fight (854 N). (Tier-1 test pins this invariance.)
- `COCARRY_REWARD_STRESS_TANH_SCALE_N = 100 N`: the f_max force scale, so the
  penalty saturates over a force range comparable to the constraint; the ~724 N
  fight excess saturates it. Bounded ⇒ safe under the normaliser.
- `COCARRY_REWARD_STRESS_COEFF = 1.0`: parity with transport/level/settle — the
  fight is penalised on the same scale cooperation is rewarded.

### 2. Transport PBRS (supporting) — `chamber.envs.cocarry_shaping`
`F = γ·Φ(s') − Φ(s)`, `Φ = −COCARRY_REWARD_TRANSPORT_PBRS_COEFF · dist(centroid, goal)`,
γ = the training MDP's discount. Ng-Harada-Russell **policy-invariant** — it
cannot change the optimum, only sharpen the gradient toward goal-directed
transport (the failure showed centroid 0.37 m off). Applied as a stateless
live-read wrapper around the training env (mirrors the Stage-1 settle PBRS),
gated on `shaping.transport_pbrs_coeff > 0` (default 0 = byte-identical for
non-co-carry cells). `COCARRY_REWARD_TRANSPORT_PBRS_COEFF = 1.0` (parity,
bounded); the config's `shaping.transport_pbrs_coeff` must equal it (Tier-1
parity test).

## What is unchanged / re-frozen

- The **success predicate** `evaluate()` is byte-untouched; every evaluation
  measures **unshaped** success.
- The **matched reference (100%)** and **f_max (130.6 N)** derive from the
  hand-written controller, which a reward edit cannot change — so **Step 1 is
  NOT re-run**; the committed `cocarry_rung2_fmax_distribution.json` is reused.
  f_max stays 130.6 N; `previous_provisional_n: 130.0` unchanged.
- The freeze enumerator captures the 4 new coefficients automatically (Tier-1
  completeness guard); re-freeze after training so the manifest is complete.

## Decision rules (pre-committed; not improvised)

1. **GPU smoke gate.** If the smoke shows **no stress drop** (p90 not trending
   toward the cooperative band) **and success still 0** → **STOP and report**;
   the reward fix is necessary-but-insufficient and the escalation
   (warm-start/curriculum, then residual-on-impedance) is a separate founder
   decision. Do not grind / tune to 100%.
2. **Train-to-reference** (only if the smoke shows the fight resolving) via
   `cocarry_rung2_train.sh`. The Step-1 stop criterion is unchanged: cannot
   reach the 100% reference (within 10 pp) / no significant slope → exit 3,
   manifest not written, **do not lower the bar**.
3. **Freeze on success**: manifest written with the new coefficients; verify
   completeness + byte-identical reload.

## Evidence
Smoke + (if reached) train artifacts and the re-frozen manifest are committed
under this directory; every reported figure links to its artifact + repro
command + seeds + commit SHA.
