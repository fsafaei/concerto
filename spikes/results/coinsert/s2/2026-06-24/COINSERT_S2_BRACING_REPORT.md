# Co-insert S2 — structured base + bracing path → SAPIEN constraint-fidelity wall (2026-06-24)

**Slice.** S2 — the structured base inserter + the cooperative reference holder, then (after the founder
S2-insertion-wall and assembly-drag rulings) the **bracing path**: cut the insertion force, then hold so the
reference *resists* the reaction. Phase-2, non-gating (ADR-026 §Decision 1-4). **Pre-S3-freeze; DO NOT MERGE.**

**Verdict.** **SAPIEN_CONSTRAINT_FIDELITY_WALL** — a sim / force-trust blocker, **not** a Gate-D task finding.
SAPIEN's drive-weld cannot represent a cooperative holder that braces the axial insertion reaction without either
over-constraining the holder arm or losing the socket orientation; the only cleanly-representable weld (a lateral
bracket) cannot brace, so the matched pair seats only part-way. Per the Stage-2 §7 escalation this triggers the
**founder decision: the pre-committed S1b MuJoCo migrate path vs an honest close.**

**Evidence.** All numbers come from the committed artifact `coinsert_s2_bracing_probe.json` (this directory).
Reproduce (GPU + oracle): `uv sync --all-extras --group dev --group oracle && uv run python
scripts/repro/_coinsert_s2_bracing_probe.py`. Seeds {0,1,2}; deterministic (env P6 substream + deterministic
controllers). No number here is hand-entered.

## What was built (the S2 instrument)

- **`CoInsertEnv`** now has the real reward + the joint success predicate `seated ∧ within_force ∧ static ∧ settled`
  (`evaluate_coinsert_success`) + per-episode peak-force accumulation (peg-socket contact + the friction-inclusive
  workpiece interaction wrench), plus the vertical top-down insertion geometry (peg welded tip-down to the ego;
  socket welded opening-up to the holder; square peg/socket yaw-aligned).
- **`chamber.partners.coinsert_impedance`** — the structured base inserter (`coinsert_base_inserter`: EE-space
  impedance approach + force-guided press + lead-in spiral / retract-repress unjam; 6-DOF orientation hold;
  per-episode integrator reset, asserted + property-tested) and the cooperative reference holder
  (`coinsert_reference_holder`), both routed through the `FrozenPartner` interface. A 2-anchor weld holds the held
  bodies' orientation (resolving the droop the founder flagged: align ~1°).
- New `PandaJacobianProvider.jacobian_6x7` / `fk_tcp_rotation` for the 6-DOF orientation hold.

## The bracing path (founder-directed) and the three findings

1. **Force cut (founder step 1).** Softening the inserter press (force-guided, not a ram) cuts the peak forces
   substantially — peak workpiece wrench **515.7 N → 92.2 N**; the reproducible ~32 mm wedge was a brute-force ram.
2. **Lateral hold drags.** Even at the reduced force, the clean-weld lateral bracket hold cannot structurally brace
   the axial reaction: the holder wrist yields to the moment, the **free** assembly drags down (~209.6 mm sink),
   and the peg seats only to **30.7 mm** (short of the 38 mm target). Friction-independent (≈33-37 mm across
   μ ∈ {0.2…0.5}).
3. **Below-hold braces but over-constrains.** A hand-below-socket hold **does** brace (socket-z holds — it resists
   the reaction in compression), but SAPIEN's drive-weld over-constrains the holder arm at **every** weld anchor
   span (zero-action wrist preload **573 N → 1306 N** as the span grows from 20 → 100 mm; socket tilt 45° → 22°),
   so the peg cannot engage.

There is no SAPIEN drive-weld hold that both braces axially **and** holds the socket orientation without
over-constraint.

## Decision rule (pre-committed) and what this is NOT

- This is **not a Gate-D STOP**: a Gate-D STOP is reserved for a clean, competent, FREE hold that still cannot seat.
  Here the hold fails to brace because of a sim/constraint-representation limit, not because cooperative insertion
  is impossible.
- This is **not resolved with active raise-to-meet mating**: founder-ruled construct-invalid (it would make the task
  require a co-designed mating choreography — the exact co-design confound this program exists to avoid — and
  conflicts with the black-box-partner premise; the accommodation burden is on the ego, and the task must be
  seatable with the holder merely bracing).
- The matched pair does not reach the ≥0.9 seat precondition, so the downstream S2 measurements (capability gate M,
  `f_insert_max` / `f_couple_max` derivation, the F-block oracle calibration) are **not** run — this slice stops at
  the force-trust blocker, as pre-committed.

## Founder decision triggered

Per Stage-2 §7 (provided for exactly "the sim can't represent the contact / constraint physics"): choose between
**(a)** the pre-committed **S1b MuJoCo migrate** path for the insertion env, or **(b)** an **honest close** of the
co-insert bet. Governance: STEP-0 tag `prereg-coinsert-s0-2026-06-24` is already cut on the S0 commit, so the
holder/controller changes are documented here (and for the future S3 freeze tag), **not** by editing the locked
`coinsert_s0.yaml`.
