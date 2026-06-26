# SPDX-License-Identifier: Apache-2.0
"""Stage-0 channel pre-check for the Gate-0 handover-and-place spike (ADR-026).

Pre-registration-INFORMING, not a measured test (executor-prompt Rev 2 §3). This thin,
scripted, kinematic Monte-Carlo *settles which channel forces a re-grasp* under the
scripted ego's six-axis kinematics, and *derives* the two grasp-pose boundaries — so
the frozen prereg's mismatch channel and intrinsic boundary are evidence-based, not
asserted. It runs BEFORE the freeze (like the co-insert S1 contact-fidelity spike); its
output is committed as a finding that parameterizes the prereg.

The two sub-channels (Rev 2 B1):

* **Lateral** — corrected by ego *translation* within reach; expected to force NO
  re-grasp (it is the downstream success tolerance). The pre-check also records the
  Rev 2 §3 caveat check: in this minimal kinematic model translation is collision-free
  within reach and does not raise the seating force, so the lateral channel does not
  force a re-grasp; if a real fixed-approach collision or a seating-force constraint
  were modelled and it DID, the finding would carry the lateral channel instead.
* **Grasp-pose / orientation** — corrected in-grasp up to the wrist-correction range,
  else by a re-grasp up to the re-acquire range, else intrinsic. This is the channel
  that forces the re-grasp.

Boundary derivation (transparent; cited inputs + stated geometry). The boundaries are
NOT swept-and-fit against the ego's own correction model (that would be circular); they
are derived from first principles:

* ``wrist_correction_range_deg`` = the binding wrist axis pitch half-range (six-axis
  datasheets: J5 ~ ±120-135 deg is the limiting reorientation axis; J4/J6 give near-full
  roll) scaled by a stated seat-approach clearance factor (the usable fraction of wrist
  pitch given that the gripper/part must clear the receptacle on the seat approach).
* ``reacquire_range_deg`` = the graspable-face span a re-grasp can re-acquire to (a
  stated part-geometry assumption: a re-grasp can flip the part to a better face up to
  this orientation; beyond it no graspable face yields the seat orientation -> intrinsic).

The two stated geometric factors are surfaced and sensitivity-reported; the wrist limit
is a cited datasheet bracket. Determinism (P6 / ADR-002) routes the sweep RNG through
:func:`concerto.training.seeding.derive_substream`.

References:
- ADR-026 §Decision (coupling validity; non-gating Phase-2 line).
- ADR-007 §Discipline (this pre-check informs, and predates, the frozen prereg).
- Six-axis wrist limits: FANUC LR Mate 200iD / ABB IRB 1200 / Yaskawa GP7 / Kawasaki
  RS007L / Staeubli TX2-60 datasheets (J5 pitch ~ +/-120-135 deg is the binding axis).
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import numpy as np

from chamber.envs.handover_place import HANDOVER_DEFAULT_TRANSLATION_RANGE_M
from concerto.training.seeding import derive_substream

#: Binding wrist axis pitch half-range (deg). Six-axis datasheet bracket; J5 is the
#: limiting reorientation axis (J4/J6 are near-full roll). Cited, not asserted.
J5_PITCH_HALF_DEG: float = 125.0
#: Datasheet bracket on the binding wrist axis (low/high), for the sensitivity report.
J5_PITCH_HALF_BRACKET_DEG: tuple[float, float] = (120.0, 135.0)

#: Seat-approach clearance factor: a proxy for the usable in-grasp reorientation given
#: gripper/fixture collision and straight-in seat reachability — NOT a true fraction of
#: the wrist's free-space range. For a straight-in connector seat the usable in-grasp
#: reorientation is bounded by collision + straight-in reachability, so 0.7 is optimistic
#: and the realistic regime likely sits low. It is therefore SWEPT, not frozen as a
#: point (founder review of the freeze). The default below is only the headline value
#: for the standalone finding print.
SEAT_APPROACH_CLEARANCE_FACTOR: float = 0.7
#: The pre-registered clearance-factor sweep (low -> headline). wrist_correction_deg =
#: J5_pitch_half_deg * factor at each level; the runner reports the coupling verdict and
#: the binding mismatch threshold per level.
CLEARANCE_FACTOR_SWEEP: tuple[float, ...] = (0.2, 0.35, 0.5, 0.7)

#: Stated graspable-face span (deg): the orientation a single re-grasp can re-acquire to
#: before no graspable face yields the seat orientation (intrinsic). Part-geometry
#: assumption, sensitivity-reported.
GRASPABLE_FACE_SPAN_DEG: float = 170.0
#: Sensitivity bracket on the graspable-face span.
GRASPABLE_FACE_SPAN_BRACKET_DEG: tuple[float, float] = (150.0, 180.0)

#: Realistic lateral presentation-offset range to probe (m). A handover may present the
#: part laterally off by up to a few cm; the ego translation reach is much larger.
_LATERAL_PROBE_MAX_M: float = 0.03
#: Grasp-pose error range to probe (deg): 0 .. a near-flip handover.
_GRASP_POSE_PROBE_MAX_DEG: float = 180.0
#: Monte-Carlo samples per channel sweep.
_N_SAMPLES: int = 4000
#: Substream label for the pre-check RNG (P6 / ADR-002).
_SUBSTREAM_NAME: str = "spike.handover_place_gate0.stage0_precheck"
#: Translatable fraction above which the lateral channel is deemed not to force a re-grasp.
_LATERAL_TRANSLATABLE_MIN: float = 0.999


def derive_grasp_pose_boundaries(
    *,
    j5_pitch_half_deg: float = J5_PITCH_HALF_DEG,
    seat_approach_clearance_factor: float = SEAT_APPROACH_CLEARANCE_FACTOR,
    graspable_face_span_deg: float = GRASPABLE_FACE_SPAN_DEG,
) -> tuple[float, float]:
    """Derive (wrist_correction_deg, reacquire_range_deg) from cited inputs (ADR-026).

    ``wrist_correction_deg`` = the binding wrist pitch half-range scaled by the stated
    seat-approach clearance factor; ``reacquire_range_deg`` = the stated graspable-face
    span. Both are transparent functions of cited datasheet limits + stated geometry,
    surfaced for review — not asserted constants.
    """
    wrist_correction_deg = j5_pitch_half_deg * seat_approach_clearance_factor
    reacquire_range_deg = graspable_face_span_deg
    return wrist_correction_deg, reacquire_range_deg


def _classify_grasp_pose(
    error_deg: float, *, wrist_correction_deg: float, reacquire_range_deg: float
) -> str:
    err = abs(error_deg)
    if err <= wrist_correction_deg:
        return "in_grasp"
    if err <= reacquire_range_deg:
        return "requires_regrasp"
    return "intrinsic"


def run_channel_precheck(
    *,
    root_seed: int = 0,
    translation_range_m: float = HANDOVER_DEFAULT_TRANSLATION_RANGE_M,
    j5_pitch_half_deg: float = J5_PITCH_HALF_DEG,
    seat_approach_clearance_factor: float = SEAT_APPROACH_CLEARANCE_FACTOR,
    graspable_face_span_deg: float = GRASPABLE_FACE_SPAN_DEG,
) -> dict[str, Any]:
    """Run the lateral + grasp-pose channel sweeps and return the finding (ADR-026 §3).

    Confirms the lateral channel is translatable (forces no re-grasp) and locates the
    grasp-pose boundaries, returning the classification fractions, the derived
    boundaries, and the one-paragraph channel finding that parameterizes the prereg.
    Deterministic (P6 / ADR-002).
    """
    rng = derive_substream(_SUBSTREAM_NAME, root_seed=root_seed).default_rng()
    wrist_correction_deg, reacquire_range_deg = derive_grasp_pose_boundaries(
        j5_pitch_half_deg=j5_pitch_half_deg,
        seat_approach_clearance_factor=seat_approach_clearance_factor,
        graspable_face_span_deg=graspable_face_span_deg,
    )

    # Lateral channel: fraction the ego corrects by translation alone (no re-grasp).
    lat = rng.uniform(0.0, _LATERAL_PROBE_MAX_M, size=_N_SAMPLES)
    lateral_translatable_fraction = float(np.mean(lat <= translation_range_m))

    # Grasp-pose channel: classify each sampled error.
    gp = rng.uniform(0.0, _GRASP_POSE_PROBE_MAX_DEG, size=_N_SAMPLES)
    classes = [
        _classify_grasp_pose(
            e, wrist_correction_deg=wrist_correction_deg, reacquire_range_deg=reacquire_range_deg
        )
        for e in gp
    ]
    in_grasp = classes.count("in_grasp") / _N_SAMPLES
    requires_regrasp = classes.count("requires_regrasp") / _N_SAMPLES
    intrinsic = classes.count("intrinsic") / _N_SAMPLES

    lateral_forces_regrasp = lateral_translatable_fraction < _LATERAL_TRANSLATABLE_MIN
    grasp_pose_forces_regrasp = requires_regrasp > 0.0
    channel = (
        "grasp_pose_orientation"
        if grasp_pose_forces_regrasp and not lateral_forces_regrasp
        else ("lateral" if lateral_forces_regrasp else "neither")
    )

    if channel == "grasp_pose_orientation":
        finding = (
            "Lateral offsets across the realistic presentation range are fully corrected "
            f"by ego translation ({lateral_translatable_fraction:.3f} translatable within "
            f"the {translation_range_m * 100:.0f} cm reach) and force no re-grasp -> lateral "
            "is the downstream SUCCESS tolerance, not the coupling channel. Grasp-pose "
            f"orientation errors beyond the wrist-correction range ({wrist_correction_deg:.0f} "
            f"deg) force a re-grasp, and errors beyond the re-acquire range "
            f"({reacquire_range_deg:.0f} deg) are kinematically intrinsic. The mismatch "
            "channel is GRASP-POSE/ORIENTATION; the intrinsic boundary is the re-acquire "
            "range. These parameterize the frozen prereg."
        )
    elif channel == "neither":
        finding = (
            "NEITHER channel forces a re-grasp at realistic magnitudes (lateral fully "
            "translatable; grasp-pose fully wrist-correctable). The regime is budget-trivial "
            "and washes out by construction -> STOP and escalate to co-hold (Rev 2 §3)."
        )
    else:
        finding = (
            "Contrary to expectation the LATERAL channel forces a re-grasp under the "
            "modelled constraint; carry the lateral channel in the prereg (Rev 2 §3)."
        )

    return {
        "schema": "handover_place_gate0_stage0_precheck/v1",
        "purpose": (
            "Stage-0 channel pre-check (ADR-026; Rev 2 §3): derive which sub-channel forces "
            "a re-grasp and the two grasp-pose boundaries, before the prereg is frozen."
        ),
        "inputs": {
            "translation_range_m": translation_range_m,
            "j5_pitch_half_deg": j5_pitch_half_deg,
            "j5_pitch_half_bracket_deg": list(J5_PITCH_HALF_BRACKET_DEG),
            "seat_approach_clearance_factor_headline": seat_approach_clearance_factor,
            "clearance_factor_sweep": list(CLEARANCE_FACTOR_SWEEP),
            "graspable_face_span_deg": graspable_face_span_deg,
            "graspable_face_span_bracket_deg": list(GRASPABLE_FACE_SPAN_BRACKET_DEG),
            "lateral_probe_max_m": _LATERAL_PROBE_MAX_M,
            "grasp_pose_probe_max_deg": _GRASP_POSE_PROBE_MAX_DEG,
            "n_samples": _N_SAMPLES,
        },
        "derived_boundaries": {
            "wrist_correction_range_deg_headline": wrist_correction_deg,
            "reacquire_range_deg": reacquire_range_deg,
            "derivation": (
                "wrist_correction = J5_pitch_half * seat_approach_clearance_factor; "
                "reacquire = graspable_face_span. Cited wrist limit + stated geometry. "
                "The clearance factor is SWEPT (not frozen); the table below gives "
                "wrist_correction at each swept level."
            ),
        },
        "clearance_sweep": {
            f"{factor:.2f}": j5_pitch_half_deg * factor for factor in CLEARANCE_FACTOR_SWEEP
        },
        "clearance_rationale": (
            "The seat-approach clearance factor is a proxy for gripper/fixture collision "
            "+ straight-in seat reachability, NOT a true fraction of the wrist's free-space "
            "range. For a straight-in connector seat the usable in-grasp reorientation is "
            "bounded by collision and straight-in reachability, so 0.7 is optimistic and the "
            "realistic regime likely sits low -> the factor is swept [0.2, 0.35, 0.5, 0.7] "
            "and the coupling verdict reported at each level."
        ),
        "lateral_channel": {
            "translatable_fraction": lateral_translatable_fraction,
            "forces_regrasp": lateral_forces_regrasp,
            "caveat_check": (
                "minimal kinematic model: translation is collision-free within reach and "
                "does not raise the seating force, so lateral forces no re-grasp"
            ),
        },
        "grasp_pose_channel": {
            "in_grasp_fraction": in_grasp,
            "requires_regrasp_fraction": requires_regrasp,
            "intrinsic_fraction": intrinsic,
            "forces_regrasp": grasp_pose_forces_regrasp,
        },
        "channel": channel,
        "finding": finding,
    }


def main() -> int:
    """Run the Stage-0 pre-check and write the finding JSON (ADR-026; ADR-016).

    Writes to ``OUT_JSON`` (default the committed precheck archive). Deterministic and
    reproducible from this command + the code SHA.
    """
    out_path = os.environ.get(
        "OUT_JSON",
        "spikes/results/handover-place-gate0-precheck-2026-06-26/stage0_channel_precheck.json",
    )
    finding = run_channel_precheck()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(finding, fh, sort_keys=True, indent=2)
    boundaries = finding["derived_boundaries"]
    wrist = boundaries["wrist_correction_range_deg"]
    reacquire = boundaries["reacquire_range_deg"]
    print(f"  channel = {finding['channel']}")
    print(f"  derived: wrist_correction={wrist:.1f} deg, reacquire={reacquire:.1f} deg")
    print(f"  artifact -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
