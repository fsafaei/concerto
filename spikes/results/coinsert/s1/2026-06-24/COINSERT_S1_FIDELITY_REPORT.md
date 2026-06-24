# Co-insert S1 — contact-fidelity spike + MuJoCo oracle (2026-06-24)

**Slice.** S1 — the make-or-break contact-fidelity check (ADR-005 §Decision; ADR-026 §Decision 4). Phase-2, non-gating.
**Verdict.** **STAY on SAPIEN** (oracle retained for the S6 re-score).
**Evidence.** All numbers come from the committed artifact
`coinsert_s1_fidelity_sweep.json` (this directory). Reproduce with the command below; no number here is hand-entered.

## What was run

With the holder scripted to a fixed pose, the peg lateral misalignment was swept across the frozen clearance set
{1.0, 0.5, 0.2} mm and the peg-socket contact force recorded on **both** simulators:

- **SAPIEN** — the env fidelity-probe rig (`make_coinsert_env(fidelity_probe=True)`): a kinematic box peg teleported
  to a controlled lateral penetration into a blind square socket that is a dynamic body rigidly anchored to a fixed
  kinematic mount; contact read via the SAPIEN contact-pair impulse (a lateral *normal* contact, so the
  friction-excluded contact-pair force — SAPIEN issue #281 — captures the lateral force faithfully; friction here is
  tangential). The friction-inclusive `get_link_incoming_joint_forces` workpiece-frame interaction-wrench instrument
  is implemented and exposed for the experiment's cooperation-cost signal (`workpiece_interaction_wrench`).
- **MuJoCo oracle** — a dimensionally-identical mirror (`scripts/repro/_coinsert_s1_mujoco_oracle.py`) with the same
  socket geometry + the same declared friction, reading the native contact force (`mj_contactForce`).

Both rigs hold the peg rigidly at a controlled lateral penetration; the declared Coulomb friction (0.5) is shared.

## Result (from the committed artifact)

| clearance | SAPIEN monotone | oracle monotone | shape-normalised divergence | raw peak ratio (SAPIEN/oracle) |
|---|---|---|---|---|
| 1.0 mm | yes | yes | 0.024 | 5.85 |
| 0.5 mm | yes | yes | 0.012 | 5.79 |
| 0.2 mm | yes | yes | 0.014 | 5.55 |

- **Monotone, graded, non-penetrating** on both simulators at every clearance — a clean ramp with contact onset at
  the geometric expectation (≈ clearance/2), **not** a cliff or penetration blow-up. This is the make-or-break
  criterion (the co-carry over-coupling-wall lesson: a numerical cliff is the disqualifier), and it passes.
- The **shape-normalised** SAPIEN and oracle curves agree to **1–2.4%** across all clearances — well inside the
  pre-registered 20% tolerance. The contact *model* is faithful in shape and onset.
- The **raw** absolute magnitudes differ by a ~5.5–5.9× factor (reported transparently as `raw_peak_ratio`). This is
  a contact-**stiffness convention** difference between PhysX (rigid) and MuJoCo, **not** a SAPIEN penetration
  pathology — both curves are physically graded. The cooperation-cost instrument is therefore calibrated on measured
  matched-pair distributions (the S2 plan), never on an absolute force threshold transferred across engines.

## Decision rule (pre-committed) and verdict

> STAY on SAPIEN iff the SAPIEN contact is monotone in misalignment (graded ramp, no cliff/penetration) **and**
> agrees with the MuJoCo oracle within tolerance; else MIGRATE (→ S1b).

Both conditions hold: SAPIEN contact is monotone/graded at every clearance, and the shape-normalised agreement with
the oracle is 1–2.4% (< 20%). **Verdict: STAY on SAPIEN**, with the MuJoCo oracle retained for the S6 re-score and
the absolute-force cross-check to be done under matched contact-compliance (the raw cross-engine magnitude offset is
a stiffness convention, surfaced here for the record).

## Reproduce

```
uv sync --all-extras --group dev --group oracle
uv run python scripts/repro/_coinsert_s1_fidelity_sweep.py
```

Deterministic: the SAPIEN probe routes RNG through the env P6 substream (`reset(seed=...)`, seeds {0,1,2}); MuJoCo's
contact solve is deterministic. Output: `coinsert_s1_fidelity_sweep.json` (this directory).
