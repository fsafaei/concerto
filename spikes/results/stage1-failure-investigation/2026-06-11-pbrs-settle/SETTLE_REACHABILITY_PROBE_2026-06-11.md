# Settle-reachability pre-flight — the 0.2 rad/s threshold is one action away

**Date.** 2026-06-11.
**Author.** Farhad Safaei.
**Status.** Phase-1 pre-flight of the PBRS-settle slice (new dated directory; I8). No
training; checkpoint replays only (A1 seeds 0/1/2, step 19,000,320, the regime-alignment
archive's policies).
**Question.** Is the `is_static` threshold (max arm |qvel| ≤ 0.2 rad/s) *physically reachable*
under the `pd_joint_delta_pos` controller while holding the placed cube — and at what settle
time from the measured ~0.5 rad/s hold? (The probe verified the predicate's semantics; nothing
had verified reachability. An unreachable threshold would make reward shaping pointless and
the question metric-level/ADR — the pre-stated STOP branch.)

## Protocol

Per hold state: roll the deterministic A1 policy to an established placed-hold (placement
lands ~step 11; switch at step 30), then command **zero deltas** on all 8 ego action dims for
k=50 steps (PD hold on current joint targets; gripper aperture retained), partner live (the
conservative case). Record per-step max arm |qvel|. 2 episodes × 3 seeds = 6 hold states.
Artifact: `settle_reachability_trajectories.json` (full per-step trajectories).

## Results

| seed / ep | placed at switch | \|qvel\| at switch | settle steps to < 0.2 | floor (median, last 10 steps) | placed throughout hold |
|---|---|---|---|---|---|
| 0 / 0 | yes | 0.617 | **0** | 0.0023 | yes |
| 0 / 1 | yes | 0.484 | **0** | 0.0038 | yes |
| 1 / 0 | yes | 0.531 | **0** | 0.0023 | yes |
| 1 / 1 | yes | 0.403 | **0** | 0.0029 | yes |
| 2 / 0 | yes | 0.609 | **0** | 0.0027 | no (cube drifted past goal_thresh mid-hold) |
| 2 / 1 | yes | 0.614 | **0** | 0.0023 | yes |

**Floor < 0.2 confirmed — overwhelmingly.** From every hold state, the *first* zero-delta
control step already reads below the threshold, and the arm settles to a floor of
0.002–0.004 rad/s (50–100× under it). Stillness while holding the placed cube is not a hard
motor-control problem under this controller; it is a single sustained action choice the
trained policies never make. One of six holds lost the `is_obj_placed` flag mid-stillness
(cube–goal drift across the 0.025 m threshold with the gripper static) — placement is not
perfectly passive at every hold pose, which mildly strengthens the case for *learned* settling
over scripted stopping, and is noted for the characterization's interpretation table.

## Design constants for the pre-statement (Phase 2)

- **Settle time ≈ 1 control step** (not the 0.2–0.7 rad/s multi-step band the scan
  conservatively assumed): the PBRS stop bonus concentrates into a single transition.
- **Floor ≈ 0.003 rad/s** → Φ at the settled state is ≈ 0 to three decimals; the potential's
  resolution near zero is immaterial.
- **Hold speeds at switch 0.40–0.62 rad/s** → cap = 0.7 rad/s covers the band with margin
  (matching the scan's hold-qvel p10–max range).

**Verdict: reachability confirmed; the STOP branch does not fire; proceed to the Phase-2
pre-statement.**

## Cross-references

`../2026-06-11-gamma-scan/GAMMA_SCAN_CHARACTERIZATION_2026-06-11.md` §3.4 (the margins this
probe sharpens); `../2026-06-10-success-static-probe/SUCCESS_STATIC_PROBE_2026-06-10.md`
(C1 mechanism; predicate semantics); ADR-007 §Stage 1b Rev 17 (regime posture); A1 archive
`../2026-06-10-regime-alignment/` (the checkpoints replayed).
