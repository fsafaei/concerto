# SPDX-License-Identifier: Apache-2.0
"""Design-time (clearance x mismatch) overlay for the handover-and-place Gate-0 spike.

Analytical, no measurement (founder freeze review, decision #2). Closes the symmetric
knob: the prereg swept the in-grasp correction limit (clearance) but pinned the mismatch
bias; this sweeps BOTH and reports, per (clearance_factor, mismatch_bias) cell, the
binding grasp-pose threshold (degrees + matched-sigma units), whether the mismatch bias
crosses it, and the realistic mismatch mass clearing it. It lets the reviewer see
whether real mismatch mass clears the threshold across the joint knob, before any
measured slice. The MEASURED coupling region in (clearance x mismatch) is produced by
the runner at PR C.

Determinism: pure analytical (no RNG). The committed numbers mirror the prereg.
"""

from __future__ import annotations

import json
import os
import sys

from chamber.spikes.handover_place_gate0.decision import clearance_mismatch_overlay

#: Cited binding wrist axis half-range (deg); mirrors the Stage-0 finding.
_J5_PITCH_HALF_DEG: float = 125.0
#: Pre-registered sweeps (mirror the prereg).
_CLEARANCE_SWEEP: list[float] = [0.2, 0.35, 0.5, 0.7]
_MISMATCH_BIAS_SWEEP_DEG: list[float] = [15.0, 30.0, 45.0]
#: Committed central angular window + presenter sigmas (mirror the prereg).
_ANGULAR_CENTRAL_DEG: float = 5.0
_MATCHED_SIGMA_DEG: float = 2.0
_MISMATCH_SIGMA_DEG: float = 10.0


def main() -> int:
    """Write the (clearance x mismatch) design overlay artifact (ADR-026; decision #2)."""
    out_path = os.environ.get(
        "OUT_JSON", "spikes/preregistration/handover_place/gate0_clearance_mismatch_overlay.json"
    )
    rows = clearance_mismatch_overlay(
        clearance_factors=_CLEARANCE_SWEEP,
        mismatch_biases_deg=_MISMATCH_BIAS_SWEEP_DEG,
        angular_window_deg=_ANGULAR_CENTRAL_DEG,
        matched_sigma_deg=_MATCHED_SIGMA_DEG,
        mismatch_sigma_deg=_MISMATCH_SIGMA_DEG,
        j5_pitch_half_deg=_J5_PITCH_HALF_DEG,
    )
    artifact = {
        "schema": "handover_place_gate0_clearance_mismatch_overlay/v1",
        "purpose": (
            "Design-time (clearance x mismatch) threshold-crossing + realistic-mass overlay "
            "(decision #2). Closes the symmetric knob; the measured coupling region is PR C."
        ),
        "design": {
            "j5_pitch_half_deg": _J5_PITCH_HALF_DEG,
            "clearance_factor_sweep": _CLEARANCE_SWEEP,
            "mismatch_bias_sweep_deg": _MISMATCH_BIAS_SWEEP_DEG,
            "angular_window_central_deg": _ANGULAR_CENTRAL_DEG,
            "matched_grasp_pose_sigma_deg": _MATCHED_SIGMA_DEG,
            "mismatched_grasp_pose_sigma_deg": _MISMATCH_SIGMA_DEG,
            "threshold_definition": "wrist_correction (= J5_half * clearance) + angular_window",
            "mass_definition": "P(|grasp_pose_error| > threshold) under Normal(bias, sigma)",
        },
        "overlay": rows,
        "interpretation": (
            "A cell is a candidate coupling regime when the mismatch bias CROSSES the binding "
            "threshold and a substantial realistic mass clears it. At high clearance (0.7, "
            "wrist ~87.5 deg) only a near-flip mismatch crosses; at low clearance (0.2, wrist "
            "~25 deg) modest mismatches (>= ~30 deg) cross and most mass clears. The measured "
            "verdict per cell follows at PR C; this overlay shows where coupling is even "
            "possible, transparently, across BOTH knobs."
        ),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, sort_keys=True, indent=2)
    for r in rows:
        print(
            f"  clearance={r['clearance_factor']:.2f} bias={r['mismatch_bias_deg']:>4.0f}deg  "
            f"threshold={r['binding_threshold_deg']:>5.1f}deg "
            f"({r['binding_threshold_sigma']:.0f}sigma)  crossed={r['threshold_crossed']!s:>5}  "
            f"mass_clearing={r['mismatch_mass_clearing']:.3f}"
        )
    print(f"  artifact -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
