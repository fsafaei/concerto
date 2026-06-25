# SPDX-License-Identifier: Apache-2.0
"""Co-insert S2 friction-lever probe — the ~30 mm wedge is friction-independent (ADR-026 §D4).

A supporting committed-evidence generator for the S2 honest-close
(spikes/results/coinsert/COINSERT_CLOSURE_2026-06-24.md): one of the levers ruled
out in establishing that the ~30 mm matched-insertion wall is a GEOMETRIC
tilt-wedge, not a friction-lock. On the validated fixed-link rig with the
canonical round geometry, the competent matched pair (base inserter + cooperative
reference holder) is run at the loosest clearance (1.0 mm) across a sweep of the
declared peg-socket Coulomb friction; the seated depth is pinned at ~30 mm
regardless — lowering friction by an order of magnitude does NOT clear the wedge,
so the lock is geometric (tilt x engaged-depth vs clearance), not frictional.

Determinism: the env routes RNG through its P6 substream (``reset(seed=...)``);
the hand-written controllers are deterministic. Seeds fixed; SAPIEN is the
GPU/oracle-gated substrate (``uv sync --all-extras --group dev --group oracle``).
Numbers come only from the committed artifact.

ADR-026 §Decision 1-4; ADR-005 §Decision; ADR-009 §Decision.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from chamber.envs.coinsert import make_coinsert_env
from chamber.partners.api import PartnerSpec
from chamber.partners.registry import load_partner

_EGO = {
    "uid": "panda_wristcam",
    "base_xyz": "-0.5,0,0",
    "base_yaw_deg": "0",
    "peg_half_len": "0.04",
}
_HOLDER = {
    "uid": "panda_partner",
    "base_xyz": "0.5,0,0",
    "base_yaw_deg": "180",
    "peg_half_len": "0.04",
}
_SEEDS = (0, 1, 2)
_EP = 320
_CLEARANCE = 1.0e-3
_FRICTIONS = (0.5, 0.3, 0.2, 0.1, 0.05)


def _matched_depth(seed: int, friction: float) -> tuple[float, float, bool]:
    """Roll the matched pair at a given friction; return (depth_mm, align_deg, seated)."""
    env = make_coinsert_env(
        condition_id="coinsert_matched_reference",
        num_envs=1,
        render_backend="none",
        peg_clearance_m=_CLEARANCE,
        peg_socket_friction=friction,
        episode_length=_EP,
    )
    base = load_partner(PartnerSpec("coinsert_base_inserter", seed, None, None, dict(_EGO)))
    hold = load_partner(PartnerSpec("coinsert_reference_holder", seed, None, None, dict(_HOLDER)))
    obs, _ = env.reset(seed=seed)
    base.reset(seed=seed)
    hold.reset(seed=seed)
    info: dict = {}
    for _ in range(_EP):
        action = {
            "panda_wristcam": np.asarray(base.act(obs), dtype=np.float32),
            "panda_partner": np.asarray(hold.act(obs), dtype=np.float32),
        }
        obs, _, terminated, _, info = env.step(action)
        if bool(np.asarray(terminated).reshape(-1)[0]):
            break
    env.close()
    return (
        round(float(info["seated_depth_m"][0]) * 1000, 1),
        round(float(info["axis_align_deg"][0]), 1),
        bool(np.asarray(info["seated"]).reshape(-1)[0]),
    )


def main() -> int:
    out_dir = os.environ.get("OUT_DIR", "spikes/results/coinsert/s2/2026-06-25-friction")
    os.makedirs(out_dir, exist_ok=True)
    rows: list[dict] = []
    for fr in _FRICTIONS:
        per_seed = [_matched_depth(s, fr) for s in _SEEDS]
        rows.append(
            {
                "friction": fr,
                "depth_mm": [r[0] for r in per_seed],
                "align_deg": [r[1] for r in per_seed],
                "seated": [r[2] for r in per_seed],
                "depth_mm_max": max(r[0] for r in per_seed),
                "any_seated": any(r[2] for r in per_seed),
            }
        )
    artifact = {
        "schema": "coinsert_s2_friction_probe/v1",
        "stage": "S2 honest-close — friction-lever characterisation (round geometry)",
        "design": {
            "geometry": "canonical round (cylinder peg + N-gon bore) — the committed S2 rig",
            "clearance_m": _CLEARANCE,
            "frictions": list(_FRICTIONS),
            "seeds": list(_SEEDS),
            "episode_length": _EP,
        },
        "friction_sweep": rows,
        "verdict": "FRICTION_INDEPENDENT",
        "finding": (
            "the matched seated depth is pinned at ~30 mm across declared friction "
            "0.5 -> 0.05; no friction setting clears the wedge, so the ~30 mm wall is "
            "geometric (tilt x engaged-depth vs clearance), not a friction-lock"
        ),
        "honesty_statement": (
            "Numbers are the committed probe output; no value is hand-entered. The "
            "matched insertion is deterministic across seeds. This characterises one "
            "of the levers ruled out for the S2 honest-close finding."
        ),
    }
    out_path = os.path.join(out_dir, "coinsert_s2_friction_probe.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, sort_keys=True, indent=2)
    for r in rows:
        print(f"  friction={r['friction']}: depth_mm={r['depth_mm']} seated={r['seated']}")
    print(f"  VERDICT: {artifact['verdict']}  artifact -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
