# SPDX-License-Identifier: Apache-2.0
r"""Handover-place scripted-ego driver for ``chamber-eval run`` (ADR-011 as amended; ADR-028).

The handover-place leaderboard rows (CPU-scale, pure-Python kinematic
env; ADR-027 §Reporting rules): scripted-ego rows ONLY — REF-SCRIPT
(``ref_script_handover_ego``, the analytic corrector), B-RND
(``random``) and B-STAT (``static``). Learned-ego handover cells are
explicitly v1.1 roadmap and are not built here.

Every cell runs at the **committed coupling-valid anchor**
(:data:`chamber.benchmarks.partner_probe.HANDOVER_PROBE_ENV_PARAMS` —
clearance factor 0.2 → wrist correction 25°, fast arm basis, takt
1.5 s → re-grasp budget 1.13 s; the CB-04 admission prereg numbers,
verbatim from the tagged Gate-0 prereg). The partner dimension is the
mismatch sweep re-expressed as leaderboard cells: only set members
inside the MEASURED coupling-valid region
(``GATE_VERDICT_REPORT_2026-06-26.md``: the clearance-0.2 family at
30°/45° grasp-pose mismatch) are leaderboard cells — the campaign
prereg pins the member list; cells outside that region are not
leaderboard cells.

Rig conventions mirror the CB-04 admission cell
(:func:`chamber.benchmarks.admission_cells.run_handover_ablated_cell`):
``initial_state_seed = seed * 1000 + episode`` (the Gate-0 pairing
convention), one presenter step then one ego step per episode,
``force_peak`` = the seating-force proxy (the task's canonical stress
channel, ADR-027 §Versioning). B-RND samples uniformly over the env's
physical command ranges (±``translation_range_m``,
±``wrist_correction_deg``, re-grasp flag ∈ {0, 1}) via a per-episode
``derive_substream`` (ADR-002 P6); B-STAT emits the zero action.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from chamber.evaluation.results import EpisodeResult

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    from numpy.typing import NDArray

    from chamber.partners.sets import PartnerMemberSpec, PartnerSetSpec

#: Substream labels recorded in handover bundle seed schedules (ADR-002 P6).
HANDOVER_EVAL_SUBSTREAM_LABELS: tuple[str, ...] = (
    "env.handover_place",
    "evaluation.bundle_bootstrap",
)

#: The scripted-ego policy ids (ADR-011 §Decision as amended).
HANDOVER_POLICY_IDS: tuple[str, ...] = ("random", "static", "ref_script_handover_ego")

#: B-RND's per-episode substream (ADR-002 P6).
_RANDOM_SUBSTREAM: str = "benchmarks.handover_eval.ego_random.{iss}"


class _RandomHandoverEgo:
    """B-RND on handover-place: uniform over the physical command ranges (ADR-011)."""

    def __init__(
        self, *, translation_range_m: float, wrist_correction_deg: float, root_seed: int
    ) -> None:
        from concerto.training.seeding import derive_substream

        self._derive = derive_substream
        self._t = float(translation_range_m)
        self._w = float(wrist_correction_deg)
        self._root_seed = root_seed
        self._rng = derive_substream(
            _RANDOM_SUBSTREAM.format(iss=0), root_seed=root_seed
        ).default_rng()

    def reset(self, *, seed: int | None = None) -> None:
        """Re-derive the per-episode substream (ADR-002 P6; ADR-011 B-RND)."""
        self._rng = self._derive(
            _RANDOM_SUBSTREAM.format(iss=int(seed or 0)), root_seed=self._root_seed
        ).default_rng()

    def act(self, obs: Mapping[str, Any]) -> NDArray[np.float64]:
        """Uniform ``[tx, ty, reorient, regrasp]`` over the env ranges (ADR-011 B-RND)."""
        del obs
        return np.array(
            [
                self._rng.uniform(-self._t, self._t),
                self._rng.uniform(-self._t, self._t),
                self._rng.uniform(-self._w, self._w),
                float(self._rng.integers(0, 2)),
            ],
            dtype=np.float64,
        )


class _StaticHandoverEgo:
    """B-STAT on handover-place: the zero action (ADR-011 §Decision as amended)."""

    def reset(self, *, seed: int | None = None) -> None:
        """No per-episode state — deterministic by construction (ADR-011 B-STAT)."""
        del seed

    def act(self, obs: Mapping[str, Any]) -> NDArray[np.float64]:
        """The zero action, regardless of observation (ADR-011 B-STAT)."""
        del obs
        return np.zeros(4, dtype=np.float64)


def run_handover_episodes_for_set(
    *,
    policy_id: str,
    set_spec: PartnerSetSpec,
    members: list[tuple[PartnerMemberSpec, dict[str, str]]],
    seeds: list[int],
    episodes_per_seed: int,
    root_seed: int = 0,
) -> tuple[dict[int, list[EpisodeResult]], list[dict[str, object]], dict[str, str]]:
    """The handover leaderboard grid: one scripted-ego row x presenters (ADR-011; ADR-028).

    Return contract mirrors
    :func:`chamber.benchmarks.bundle_runner.run_task_episodes_for_set`
    (member-unique ``episode_idx``; matched ``iss = seed*1000 + episode``
    draws across members; redacted partner material + custody hashes).

    Raises:
        KeyError: Unknown policy id (lists the known ids).
        ValueError: Empty ``seeds`` / ``members``.
    """
    from chamber.agents.handover_ego_scripted import ScriptedHandoverEgo
    from chamber.benchmarks.partner_probe import (
        HANDOVER_PROBE_ENV_PARAMS,
        member_material,
    )
    from chamber.envs.handover_place import make_handover_place_env
    from chamber.partners.registry import load_partner

    if not seeds or not members:
        msg = "seeds and members must be non-empty"
        raise ValueError(msg)
    if policy_id not in HANDOVER_POLICY_IDS:
        known = ", ".join(HANDOVER_POLICY_IDS)
        msg = f"unknown handover policy id {policy_id!r}; known: {known}"
        raise KeyError(msg)

    p = HANDOVER_PROBE_ENV_PARAMS
    env = make_handover_place_env(
        lateral_window_m=p["lateral_window_m"],
        angular_window_deg=p["angular_window_deg"],
        seating_force_limit_n=p["seating_force_limit_n"],
        translation_range_m=p["translation_range_m"],
        wrist_correction_deg=p["wrist_correction_deg"],
        reacquire_range_deg=p["reacquire_range_deg"],
        contact_stiffness_n_per_m=p["contact_stiffness_n_per_m"],
        angular_stiffness_n_per_deg=p["angular_stiffness_n_per_deg"],
        regrasp_budget_s=p["regrasp_budget_s"],
        regrasp_duration_s=p["regrasp_duration_s"],
    )
    if policy_id == "ref_script_handover_ego":
        ego: Any = ScriptedHandoverEgo(
            translation_range_m=env.translation_range_m,
            wrist_correction_deg=env.wrist_correction_deg,
        )
    elif policy_id == "random":
        ego = _RandomHandoverEgo(
            translation_range_m=env.translation_range_m,
            wrist_correction_deg=env.wrist_correction_deg,
            root_seed=root_seed,
        )
    else:
        ego = _StaticHandoverEgo()

    episodes_by_seed: dict[int, list[EpisodeResult]] = {}
    material: list[dict[str, object]] = []
    hashes: dict[str, str] = {}
    for cluster_seed in seeds:
        records: list[EpisodeResult] = []
        for member_index, (member, params) in enumerate(members):
            variant = (
                "matched" if float(params.get("grasp_pose_bias_deg", "0")) == 0.0 else "mismatched"
            )
            member_spec = member.partner_spec(params=params, seat_extra={"variant": variant})
            presenter = load_partner(member_spec)
            if member.member_name not in hashes:
                material.append(
                    member_material(member, member_spec, redact=member.split == "private")
                )
                hashes[member.member_name] = member_spec.partner_id
            for episode in range(episodes_per_seed):
                unique_idx = member_index * episodes_per_seed + episode
                iss = int(cluster_seed) * 1000 + episode  # the Gate-0 pairing convention
                obs, _ = env.reset(seed=iss)
                presenter.reset(seed=iss)
                ego.reset(seed=iss)
                obs, _, _, _, _ = env.step(presenter.act(obs))
                _, _, _, _, info = env.step(ego.act(obs))
                records.append(
                    EpisodeResult(
                        seed=int(cluster_seed),
                        episode_idx=unique_idx,
                        initial_state_seed=iss,
                        success=bool(info["success"]),
                        force_peak=float(info["seating_force_proxy_n"]),
                        metadata={
                            "condition": "handover_place_anchor_clr0.2_fast_takt1.5",
                            "policy": policy_id,
                            "member": member.member_name,
                            "partner_set": set_spec.slug,
                            "residual_lateral_m": float(info["residual_lateral_m"]),
                            "residual_angular_deg": float(info["residual_angular_deg"]),
                            "failure_mode": str(info["failure_mode"]),
                        },
                    )
                )
        episodes_by_seed[int(cluster_seed)] = records
    return episodes_by_seed, material, hashes


__all__ = [
    "HANDOVER_EVAL_SUBSTREAM_LABELS",
    "HANDOVER_POLICY_IDS",
    "run_handover_episodes_for_set",
]
