# Co-insert (contact-rich hold-and-insert) — Phase-2 bet: honest close at HARD_STOP (2026-06-24)

**Status: CLOSED — honest close at the pre-committed HARD_STOP.** Phase-2, NON-GATING research bet (invariant I1;
ADR-026 §Decision 4). This report is the board-ready closure log: a self-contained, hostile-reviewer-auditable
record of the S0→S1→S2 arc, the decisive results, the three banked findings, the honest scoping, and the parked
successor. Every number cited comes from a committed artifact (path + JSON + repro command + commit SHA); no value
is hand-entered.

## Question and scope

The bounded Phase-2 bet: *does pairing un-co-designed robots break cooperation in a regime that does not forgive
mismatch?* — an **existence test** in a contact-rich, two-robot **hold-and-insert** task (ego = inserter trains a
residual; partner = black-box holder of a **FREE** receptacle). The bet was deliberately the unforgiving complement
to the co-carry (forgiving) ladder. It is non-gating: it touches no Phase-1 gate and no Stage-1 line. The discipline:
a heterogeneity axis is informative only if the manipulated heterogeneity is coupled to the task outcome through the
cooperation the task demands (ADR-026 §Decision 1); the difficulty knob is **clearance** (a monotone physical
parameter), not an inter-robot stiffness (the structural escape from the co-carry over-coupling wall).

## The S0 → S1 → S2 arc (decisive results, each tied to its committed artifact)

| slice | result | artifact (commit) |
|---|---|---|
| **S0** scaffold + pre-registration | env skeleton, doc-form prereg, power sim; tag `prereg-coinsert-s0-2026-06-24` cut on `729c41d` (blob `4bb9866…`) | `spikes/preregistration/coinsert/coinsert_s0.yaml` (#256, `729c41d`) |
| **S1** contact-fidelity spike | **STAY** on SAPIEN: contact monotone + graded on both sims, shape-normalised SAPIEN↔MuJoCo agreement within tolerance (~1–2.4%) | `spikes/results/coinsert/s1/2026-06-24/coinsert_s1_fidelity_sweep.json` (#257, `d81726b`) |
| **S2** base inserter + the apparent "SAPIEN wall" | `create_drive` socket weld over-constrains under axial bracing load: zero-action holder-wrist preload **573→1306 N**; lateral hold drags the free assembly **~210 mm**; force-cut peak couple **515.7→92.2 N** | `spikes/results/coinsert/s2/2026-06-24/coinsert_s2_bracing_probe.json` (`20fbace`) |
| **S2** fixed-link attach — **wall DISPROVEN** | socket as a rigid child link of the holder articulation collapses the preload **573–1306 N → 0.2 N**, braces (sink −1.4 mm), holds align ~0.7°, single-inserter ≈0 — the "wall" was a maximal-coordinate `create_drive` artifact | `spikes/results/coinsert/s2/2026-06-24-fixedlink/coinsert_s2_fixedlink_probe.json` (`582992b`) |
| **S2** bounded competence-tuning — **BOUND_HIT** | two decisive control fixes (6-DOF ori-hold 6→2.5; long-bracket fixed-link arm clearance); competent matched pair reaches **~30.3 mm**, align 0.9°, but cannot clear the last ~8 mm to the 38 mm seated threshold | `spikes/results/coinsert/s2/2026-06-25-fixedlink-tuning/coinsert_s2_fixedlink_tuning_probe.json` (`58f3da3`) |
| **S2** round-geometry Gate-D sweep — **HARD_STOP** | (i) square@1.0 mm walls (~30.3 mm, seat 0); (ii) canonical round (cylinder peg + 12-facet bore) walls at the **same ~30 mm** at every clearance (1.0/0.5/0.2 mm → matched **0/5**, rigid-hold 0/5, single-inserter 0) | `spikes/results/coinsert/s2/2026-06-25-round/coinsert_s2_round_sweep.json` (`aaf8694`) |
| **S2** friction-lever probe | matched seated depth pinned (~20–31 mm) across declared friction **0.5→0.05**, **none seated** — the wedge is geometric, not a friction-lock | `spikes/results/coinsert/s2/2026-06-25-friction/coinsert_s2_friction_probe.json` (this commit) |

Repro (GPU + oracle): `uv sync --all-extras --group dev --group oracle` then run the named
`scripts/repro/_coinsert_s2_*.py` / `_coinsert_s1_fidelity_sweep.py` generator; seeds {0,1,2} (or 0–4 for the
round sweep); deterministic (env P6 substream + deterministic controllers).

## The three banked findings

1. **The SAPIEN "constraint-fidelity wall" was a `create_drive` artifact — disproven.** Modeling the held socket as
   a **fixed child link of the holder articulation** (reduced-coordinate solver) instead of a maximal-coordinate
   `create_drive` inter-body weld collapses the over-constraint preload **573–1306 N → 0.2 N** and braces the axial
   reaction (assembly sink **−1.4 mm** vs −210 mm). Migrate-or-close on sim-fidelity grounds was correctly taken off
   the table.
2. **A competent, reusable rig + controllers.** `CoInsertEnv` (the fixed-link rig, vertical top-down insertion, the
   real `seated ∧ within_force ∧ static ∧ settled` predicate, the friction-inclusive workpiece-wrench instrument),
   plus `chamber.partners.coinsert_impedance` (the structured **base inserter** with the pre-registered insertion
   envelope, 6-DOF orientation hold, and per-episode-reset assertion; the **cooperative reference holder**). The
   matched pair is a competent inserter (reaches ~30 mm, align 0.9°, braces, single-inserter ≈0).
3. **The geometric no-operating-point finding (the close).** There is **no clearance** in the frozen set — square or
   canonical round — at which the competent matched pair seats. The ~30 mm wall is a geometric **tilt-wedge**:
   reaching the **38 mm seated threshold** (the 40 mm nominal depth target minus the 2 mm tolerance) at 0.5 mm-per-side
   clearance requires the relative peg-bore tilt held **<0.7°** through the full
   insertion, but the insertion contact itself cocks the peg to **~0.9–2.8°**, which two-point-wedges at ~30 mm. It
   is robust to **every** lever probed: architecture (`create_drive` AND fixed-link), friction (0.5→0.05),
   cross-section (square AND round), and holder rotational compliance (ori-hold 2.5→1.0 walls; ≤0.5 the socket flops
   55–71°). The **seatable region (tilt <0.7°)** and the **achievable-control region (~0.9–2.8° under contact)** do
   not overlap.

## Finding (3) stated precisely (read this before paraphrasing)

There is **no operating point** — no clearance, square or round — at which the competent matched pair seats the peg.
Because the **cooperative** hold itself cannot seat it, **no operating point ever existed at which to test
heterogeneity at all**. The rigid-hold coupling check is **vacuous** here (a rigid hold also fails to seat — but so
does the cooperative one), so this is **not** "heterogeneity would bite but we couldn't reach it," and **not** "the
holder doesn't matter." We **never reached a seating point** at which a cooperative and a heterogeneous hold could be
compared. The close is: the seatable region and the coupling-valid region do not overlap in this regime.

## Honest scoping / anticipated critiques (pre-empting the review board)

This does **not** claim that cooperative contact-rich insertion is impossible, nor that the heterogeneity thesis is
false. The finding is scoped to: **this sim** (SAPIEN/PhysX, the validated contact model — S1 STAY); a
**hand-built-impedance** base inserter (the tilt-precision wall is a control-class limit of hand-built impedance — a
**trained** residual inserter, or a different contact engine, might hold tighter tilt; untested here, by the
pre-committed bound); and a **free-receptacle 6-DOF** geometry (the receptacle is held only by the partner, the
two-robot-necessity choice). The refused weaken-to-pass moves are recorded below precisely because crossing them
would have manufactured a pass without testing the thesis.

## Meta-pattern across the two regimes (suggestive, not conclusive)

Across two deliberately opposite designs — **co-carry** (forgiving: the competent matched pair solved the task, but
capability-matched holder heterogeneity did not produce a qualifying ≥20 pp drop) and **co-insert** (unforgiving: the
competent matched pair cannot even seat) — **neither produced an operating point that is both solvable-by-a-
competent-pair AND coupling-valid.** A clean in-sim existence test (a regime where a competent pair succeeds AND
holder heterogeneity is demonstrably load-bearing) has eluded both. This is **suggestive, not conclusive** (two
hand-built-controller points, two sims-of-one-engine). It **strengthens the hybrid reframe**: rest the program's
external claim on the **runtime + the coupling-validity benchmark + the method** (the disciplined search, the
falsifiable controls, the honest stops), **not** on a clean existence result that neither regime has yielded.

## Decision (pre-committed)

**HARD_STOP / honest close**, per the founder's pre-committed (i)→(ii)→(iii) tree. The following weaken-to-pass /
construct-invalid moves were available and were **refused** (each would manufacture a pass without testing the
thesis): (a) reduce the 40 mm nominal seat-depth target; (b) loosen clearance beyond the frozen {1.0, 0.5, 0.2} mm set; (c) active
raise-to-meet mating (makes the task require co-designed choreography — the exact co-design confound). The locked
`coinsert_s0.yaml` is untouched; the square→round cross-section change is documented here (the peg cross-section is
not in the frozen prereg set; `peg_diameter`/`clearance`/`depth`/`chamfer`/`mass`/`friction`/`control`/`horizon`
are). The S1 fidelity-probe rig stays **square**; its STAY verdict stands.

## Parked successor (documented, NOT launched)

**(A) The fixtured / constrained-receptacle variant.** Replacing the free 6-DOF receptacle with a partially-fixtured
or constrained one changes the cooperation model from *bracing/coordination* to *presentation/stabilisation* — more
realistic for many assembly cells, and it may open a **solvable ∩ coupling-valid** window the free-receptacle 6-DOF
geometry does not (the tilt-precision burden shifts off the inserter). This is the candidate failure-informed
successor; it is **parked**, not launched, and would be a fresh pre-registration (a different cooperation model, not
a tune-to-pass of this one).

## Honesty statement

All numbers above are read from the committed JSON artifacts named in the arc table; none is hand-entered. The
matched insertion is deterministic across seeds (the per-episode goal jitter does not perturb the peg/socket
warm-start), so seat rates are cleanly 0 at every clearance. No co-insert episode reached the 38 mm seated threshold, so no
heterogeneity / capability-gate / force-limit / oracle-calibration measurement was run — this bet closes at the
pre-committed HARD_STOP, before any heterogeneity comparison.
