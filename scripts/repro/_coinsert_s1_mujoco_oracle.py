# SPDX-License-Identifier: Apache-2.0
"""MuJoCo contact-fidelity oracle mirror for the co-insert S1 spike (ADR-005; ADR-026 §D4).

The high-fidelity second line for the S1 contact-fidelity check: a MuJoCo
mirror of the co-insert peg-socket geometry with a **native contact-force
sensor** (``mj_contactForce``), used to validate that the SAPIEN insertion
contact is monotone in misalignment and agrees within tolerance. MuJoCo is the
contact engine the tooling evaluation selected as the dual-sim oracle; it is a
dev-only tool (the ``oracle`` dependency group), never imported by the
``chamber`` / ``concerto`` runtime surface — only by the committed S1/S6 repro
generators.

The geometry is derived from the SAME frozen constants the SAPIEN env uses
(:mod:`chamber.envs.coinsert`): a square blind socket (four walls + a floor)
of inner half-width :func:`chamber.envs.coinsert.coinsert_socket_inner_half_width`
around a box peg of half-width ``peg_diameter/2``, with the SAME declared
Coulomb friction. The peg is a MuJoCo mocap body teleported to a controlled
lateral misalignment into the socket wall — the identical protocol to the SAPIEN
kinematic-peg probe — so the two sims are compared geometry-for-geometry.

Determinism: MuJoCo's contact solve is deterministic; no RNG is used here.
"""

from __future__ import annotations

import numpy as np

from chamber.envs.coinsert import (
    COINSERT_PEG_DIAMETER_M,
    COINSERT_PEG_SOCKET_FRICTION,
    COINSERT_SOCKET_DEPTH_M,
    COINSERT_SOCKET_OUTER_HALF_M,
    coinsert_socket_inner_half_width,
)

#: Socket opening-plane height in the oracle world frame (matches the SAPIEN
#: probe socket z so the two rigs are dimensionally identical).
_Z_OPEN = 0.30

#: Floor slab half-thickness (matches the SAPIEN socket floor).
_T_FLOOR_HALF = 0.005

#: Probe peg half-length (matches the SAPIEN probe peg: ~0.4x the cavity depth).
_PEG_HALF_LEN = 0.4 * COINSERT_SOCKET_DEPTH_M


def build_mjcf(clearance_m: float, *, friction: float = COINSERT_PEG_SOCKET_FRICTION) -> str:
    """MJCF for the peg-socket oracle at a given clearance (ADR-005; ADR-026 §D4).

    Four static walls + a floor enclose a square cavity of inner half-width
    ``coinsert_socket_inner_half_width(clearance_m)``; a box peg on a mocap body
    is the kinematic mover. Friction is the frozen co-insert coefficient. Pure
    string builder — no MuJoCo import — so it is testable without the engine.

    Args:
        clearance_m: Diametral clearance (hole - peg), metres.
        friction: Coulomb friction coefficient (default the frozen co-insert value).

    Returns:
        The MJCF XML string.
    """
    r = COINSERT_PEG_DIAMETER_M / 2.0
    w_in = coinsert_socket_inner_half_width(clearance_m)
    w_out = COINSERT_SOCKET_OUTER_HALF_M
    depth = COINSERT_SOCKET_DEPTH_M
    wall = (w_out - w_in) / 2.0
    zc = _Z_OPEN - depth / 2.0  # wall centre z
    fr = f"{friction} 0.005 0.0001"  # sliding / torsional / rolling friction

    def box(name: str, sx: float, sy: float, sz: float, px: float, py: float, pz: float) -> str:
        return (
            f'<geom name="{name}" type="box" size="{sx} {sy} {sz}" '
            f'pos="{px} {py} {pz}" friction="{fr}" condim="3"/>'
        )

    # The socket is STATIC (worldbody geoms — held fixed by construction); the
    # peg is a DYNAMIC free body position-controlled by a stiff mocap weld, so it
    # has DOF (MuJoCo computes contact against the static walls) yet tracks the
    # commanded pose and deflects physically on wall contact. This mirrors the
    # SAPIEN rig (a fixed socket + a controlled peg). zc is the absolute wall
    # centre (used for both engines' geometry parity).
    # Stiff contact (small solref time-constant, high solimp) so the contact is
    # near-rigid — comparable to SAPIEN's rigid PhysX contact, so the magnitude
    # cross-check is apples-to-apples rather than measuring a soft-contact
    # convention difference.
    sol = 'solref="0.0008 1" solimp="0.999 0.9999 0.00001"'
    walls = "\n      ".join(
        [
            box("floor", w_out, w_out, _T_FLOOR_HALF, 0.0, 0.0, _Z_OPEN - depth - _T_FLOOR_HALF),
            box("wall_px", wall, w_out, depth / 2.0, w_in + wall, 0.0, zc),
            box("wall_nx", wall, w_out, depth / 2.0, -(w_in + wall), 0.0, zc),
            box("wall_py", w_in, wall, depth / 2.0, 0.0, w_in + wall, zc),
            box("wall_ny", w_in, wall, depth / 2.0, 0.0, -(w_in + wall), zc),
        ]
    )
    return f"""<mujoco model="coinsert_oracle">
  <option timestep="0.001" integrator="implicitfast" cone="elliptic"/>
  <default>
    <geom {sol}/>
  </default>
  <worldbody>
      {walls}
      <body name="peg" pos="0 0 {_Z_OPEN + 0.10}">
        <freejoint/>
        <geom name="peg" type="box" size="{r} {r} {_PEG_HALF_LEN}" density="8000"
              friction="{fr}" condim="3"/>
      </body>
  </worldbody>
</mujoco>
"""


def oracle_contact_force(
    clearance_m: float,
    misalignment_m: float,
    *,
    insertion_depth_m: float = 0.025,
    settle_steps: int = 80,
    avg_steps: int = 30,
) -> float:
    """Peak peg-socket contact force in the MuJoCo oracle, N (ADR-005; ADR-026 §D4).

    Teleports the mocap peg to a lateral ``misalignment_m`` at ``insertion_depth_m``
    below the opening, steps to settle, and returns the settled-mean total
    contact-force magnitude (summed over peg-socket contacts via
    ``mj_contactForce``). The native MuJoCo contact force is friction-inclusive
    by construction — the high-fidelity reference the SAPIEN signal is checked
    against.

    Args:
        clearance_m: Diametral clearance (hole - peg), metres.
        misalignment_m: Lateral peg offset from the socket axis, metres.
        insertion_depth_m: Peg-centre depth below the opening, metres.
        settle_steps: Sim steps after teleport.
        avg_steps: Trailing steps averaged for the settled force.

    Returns:
        The settled-mean contact-force magnitude, Newtons.
    """
    import mujoco  # noqa: PLC0415 - lazy: mujoco is the dev-only oracle dependency group

    model = mujoco.MjModel.from_xml_string(build_mjcf(clearance_m))
    data = mujoco.MjData(model)
    peg_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "peg")
    qpos = np.array(
        [misalignment_m, 0.0, _Z_OPEN - insertion_depth_m, 1.0, 0.0, 0.0, 0.0], dtype=np.float64
    )
    # Hold the peg RIGIDLY at the commanded penetration each step (kinematic-
    # equivalent — the SAPIEN kinematic-peg rig mirrored): reset qpos/qvel before
    # each solve so the free body cannot back off the wall, and read the
    # near-rigid contact constraint force MuJoCo computes to resist penetration.
    forces: list[float] = []
    buf = np.zeros(6, dtype=np.float64)
    for i in range(settle_steps):
        data.qpos[:] = qpos
        data.qvel[:] = 0.0
        mujoco.mj_step(model, data)
        if i >= settle_steps - avg_steps:
            total = 0.0
            for c in range(data.ncon):
                con = data.contact[c]
                if peg_gid in (con.geom1, con.geom2):
                    mujoco.mj_contactForce(model, data, c, buf)
                    total += float(np.linalg.norm(buf[:3]))
            forces.append(total)
    return float(np.mean(forces)) if forces else 0.0
