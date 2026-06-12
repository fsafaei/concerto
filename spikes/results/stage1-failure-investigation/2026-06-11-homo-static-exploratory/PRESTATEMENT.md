# EXPLORATORY homo-static pre-statement — does freezing the partner move homo into the hetero band?

**Date.** 2026-06-11.
**Author.** Farhad Safaei.
**Status.** DRAFT — awaiting founder sign-off. Frozen at launch (approval recorded as a new
file per the approval-record pattern; I8 from launch onward).

**EXPLORATORY — the labelling discipline, first and binding on every artifact:** this slice
is **outside the prereg'd AS conditions**. It is not a gate cell, not gate evidence, and it
cannot change, re-run, or re-frame the recorded AS verdict
(`spikes/results/stage1-AS-2026-06-11/`, **exit 5 — immutable**). Its only output is
interpretation evidence for the consultation (reading (a) vs (b);
`CONSULTATION_BRIEF_GATE_VERDICT_2026-06-11.md` §2). Every results JSON carries
`"EXPLORATORY"` in its arm string; every file and commit message carries the label; the
characterization opens with this clause.

## §1 — The single question

Does making the homo partner **static** move homo settle-conversion into the hetero band?
This isolates the #230 mis-port's *perpetual partner motion* (the homo `panda_partner` arm
moves through the shared workspace all episode; the hetero `fetch` is an empirical statue —
`AS_HOMO_OPERATIONALIZATION_2026-06-11.md` §2.1, measured) as the candidate explanation for
the gate's sign reversal.

**Mechanism note (founder addition, recorded at sign-off):** the partner-freeze removes
**both candidate disturbance paths simultaneously** — (i) physical perturbation of the
shared workspace, and (ii) partner-motion in the ego's observation vector (the partner's
qpos/qvel are part of the ego's widened state, Rev 12, so a moving partner injects
non-stationary observation features even without contact). A positive result therefore
establishes *the moving partner* as the cause **without distinguishing the path**; the
physical-vs-observational decomposition is named as a **consultation-level follow-up, not
run here**.

## §2 — Implementation: the structurally-gated partner-static knob

- **Config:** a new `ExploratoryConfig` block on `EgoAHTConfig` with a single field,
  `partner_static_override: bool = False` (Pydantic-validated, frozen, default-off →
  byte-identical pre-existing behaviour, ADR-002). YAML key
  `exploratory.partner_static_override`, documented as EXPLORATORY in the yaml comment.
- **Effect when ON:** `run_training` wraps the built partner in a zero-action override
  (emits `np.zeros(action_dim)` from `act()`, preserves the `FrozenPartner` Protocol and
  spec; the **prereg'd partner classes and the registry are untouched**). The partner stands
  at the Rev 16 ready pose, motionless — equalising the homo partner's *behaviour* to the
  hetero partner's measured behaviour while keeping the homo *embodiment*.
- **Structural gate (the safety-loud-fail pattern, not a convention):**
  1. `TrainedPolicyFactory.__init__` **raises** if
     `cfg.exploratory.partner_static_override` is true — every gate-facing run reaches
     training through the factory (`chamber-spike run --sub-stage 1b` dispatch,
     `stage1_as.py:221-226`), so the production gate path **cannot** train with the knob on,
     by construction. Error message cites this pre-statement and "EXPLORATORY only; ADR
     required for any gate-facing use."
  2. The override wrapper stamps `"exploratory_partner_static": true` into the run's JSONL
     `extra` (and thereby every log line + the run_id fingerprint via the config hash), so
     an archive can never silently contain a static-partner run.
  3. Tier-1 pins for both: factory refusal; default-off identity (no wrapper constructed at
     `False`).
- **Eval:** this slice's cold eval uses the same zero-action partner (the probe's
  frozen-partner pattern), so training and eval see the same partner behaviour — matching
  how the gate's hetero cells experienced their (empirically static) partner in both phases.
- The knob's code lands by its own EXPLORATORY-labelled `feat` PR after sign-off, before
  launch; no other code changes.

## §3 — Arms (frozen at sign-off)

| arm | embodiment | partner behaviour | seeds | regime |
|---|---|---|---|---|
| **X-homo-static (EXPLORATORY)** | AS-homo (panda + panda_partner) | static (zero-action override) | 0/1/2 | the gate regime verbatim: N=1024, γ=0.8, 20M frames, `settle_alpha=0.5`, filter-off (Rev 17/18) |

Budget ≈ 3 × ~33 min ≈ **1.7 h** chained; halt-without-retry; no mid-run changes; **no
repository mutations once the chain starts** (the #227 lesson, now a launcher-documented
rule, PR #228). Anchors (no new runs): the gate archive's homo and hetero cells.

## §4 — Instrument

Cold deterministic eval, **30 episodes/seed** (prereg'd Rev 16 init, forced cold, unshaped
success, unfiltered, `num_envs=1`), with the **W-B occupancy instrument**: per-step
placed/static/placed∧static logging; `P(static∧placed | placed)` and `P(placed | static)`
per cell; hold-qvel band (median/p10/frac<0.2). Training-side ladders + PPO health.
Fingerprinted run_ids (which differ from all gate cells by construction — the exploratory
flag is in the config hash); artifacts as new files in this directory (I8); SHA256SUMS.

## §5 — Decision rule (frozen at sign-off; interpretation-only by construction)

Reference bands from the gate archive (`spikes/results/stage1-AS-2026-06-11/`):
**homo-as-run 1/100 (per-seed 0,0,0,1,0); hetero 12/100 (per-seed 3,7,0,2,0).**

- **Moving-partner explains the reversal:** X-homo-static pooled cold success lands **in or
  above the hetero band** (pooled rate ≥ 12 %, i.e. ≥ 11/90) **with ≥ 2/3 seeds nonzero**.
- **Does not explain:** X-homo-static stays at the homo-as-run floor (pooled ≈ 1 %, i.e.
  ≤ 2/90) — the reversal then rests on embodiment/crowding-as-genuine-signal or variance,
  and the consultation carries that (this cell cannot separate arm-*presence* from
  selection-pressure asymmetry — pre-stated limit).
- **Anything between: reported as measured, no forced call.**

**Neither branch alters the recorded verdict, any prereg artifact, the register, or any gate
archive.** The output feeds the consultation reply (and the #230 classification question)
only.

## Cross-references

`../2026-06-11-gate-verdict/` (operationalization audit; consultation brief; the skeleton
this completes); `spikes/results/stage1-AS-2026-06-11/` (the immutable verdict + reference
bands); issues #230 (the mis-port; classification pending), #233 (prereg-notes
discrepancy); ADR-007 Revs 12 (widened state — the observational path), 13 (the zero-action
ablation precedent), 15–18 (rig + regime contracts); ADR-002 (default-off byte-identity);
ADR-009 (the partner Protocol the override preserves).
