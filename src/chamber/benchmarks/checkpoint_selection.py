# SPDX-License-Identifier: Apache-2.0
r"""Preregistered checkpoint selection for learned baselines (ADR-027 §Reporting rules).

ADR-027 §Reporting rules: *"Checkpoint selection for learned baselines
is preregistered: per seed, the checkpoint with highest stress-compliant
success on a held-out validation partner."* This module is that rule as
a utility, generalising the committed Rung-2 precedent
(``spikes/results/cocarry/rung2/cocarry_rung2_freeze_selection_prereg.json``
+ its validation companion) — the defined, defensible measurement that
converts the Rung-2 "success wandered between checkpoints" failure mode
into a rule instead of a judgement call.

The rule, exactly:

1. Candidates are the run's saved checkpoints (step-ordered).
2. Each candidate is evaluated on the **held-out validation partner**
   (a public set member the campaign prereg names and excludes from
   every eval cell) over the preregistered selection seed grid.
3. The score is **stress-compliant success**: episode success AND the
   episode's stress/tilt maxima inside the task limits
   (:data:`chamber.envs.cocarry.COCARRY_STRESS_MAX_PROXY_N` /
   :data:`~chamber.envs.cocarry.COCARRY_TILT_MAX_DEG`). The co-carry
   joint-success predicate already embeds the unstressed conjunct; the
   explicit conjunction is kept so the rule generalises to tasks whose
   success predicate does not, and so the artifact records both counts.
4. The selected checkpoint maximises the score; **ties break to the
   earlier step** (the Rung-2 "earliest constraint-clean checkpoint at
   the max clean success" precedent).
5. Selection happens **before any eval-cell episode is seen** — the
   caller's procedural obligation, stated in the campaign prereg;
   the artifact this module emits is the evidence trail.

B-JOINT (ADR-011 §Decision as amended) is evaluated as the pair it
trained as, so its candidates are scored in ``pair`` mode: the partner
seat is the same pair checkpoint's partner-side actor, not a set
member (its "validation partner" is definitionally its own).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path

    from chamber.evaluation.results import EpisodeResult
    from chamber.partners.sets import PartnerMemberSpec, PartnerSetSpec

#: Selection-artifact schema id (the Rung-2 ``.../v1`` convention).
SELECTION_SCHEMA: str = "cocarry_baselines_checkpoint_selection/v1"

#: The tie-break + maximisation rule, stated once and recorded verbatim
#: in every artifact (ADR-027 §Reporting rules; Rung-2 precedent).
SELECTION_RULE: str = (
    "per seed: highest stress-compliant success on the held-out validation "
    "partner over the preregistered selection seeds; ties break to the "
    "earliest checkpoint step"
)


@dataclass(frozen=True)
class CheckpointCandidate:
    """One saved checkpoint of one training seed (ADR-027 §Reporting rules).

    Attributes:
        step: Training step the checkpoint was saved at.
        uri: The ``local://artifacts/...`` checkpoint URI.
    """

    step: int
    uri: str


@dataclass(frozen=True)
class CandidateScore:
    """Per-candidate selection evidence (ADR-027 §Reporting rules; Rung-2 precedent).

    Attributes:
        step: The candidate's training step.
        uri: The candidate's checkpoint URI.
        n_episodes: Episodes evaluated.
        successes: Raw joint-success count.
        compliant_successes: Stress-compliant success count (the score).
        stress_p90: 90th percentile of the episode stress maxima (N).
        stress_max: Max episode stress (N).
        tilt_max_deg: Max episode tilt (degrees).
    """

    step: int
    uri: str
    n_episodes: int
    successes: int
    compliant_successes: int
    stress_p90: float
    stress_max: float
    tilt_max_deg: float


@dataclass(frozen=True)
class CheckpointSelectionResult:
    """The selection outcome + full evidence table (ADR-027 §Reporting rules).

    Attributes:
        schema: :data:`SELECTION_SCHEMA`.
        selection_rule: :data:`SELECTION_RULE`, verbatim.
        policy_prefix: The candidate policy kind (``happo`` /
            ``happo_blind`` / ``joint_ego``).
        validation_partner: The held-out member's name (``"<pair>"``
            in pair mode — B-JOINT's partner is its own).
        selection_seeds: The preregistered selection seed grid.
        episodes_per_seed: Episodes per selection seed.
        scores: Step-ordered per-candidate evidence rows.
        selected_step: The winning step.
        selected_uri: The winning checkpoint URI.
    """

    schema: str
    selection_rule: str
    policy_prefix: str
    validation_partner: str
    selection_seeds: list[int]
    episodes_per_seed: int
    scores: list[CandidateScore]
    selected_step: int
    selected_uri: str


def score_episodes(step: int, uri: str, episodes: list[EpisodeResult]) -> CandidateScore:
    """Aggregate one candidate's episodes into its evidence row (ADR-027 §Reporting rules).

    Stress-compliant success = ``success`` AND ``force_peak`` ≤ the
    task stress limit AND the recorded episode tilt maximum ≤ the tilt
    limit (limits from :mod:`chamber.envs.cocarry`; the co-carry
    success predicate already embeds the unstressed conjunct — both
    counts are recorded).
    """
    import numpy as np

    from chamber.envs.cocarry import COCARRY_STRESS_MAX_PROXY_N, COCARRY_TILT_MAX_DEG

    successes = 0
    compliant = 0
    stresses: list[float] = []
    tilts: list[float] = []
    for ep in episodes:
        stress = float(ep.force_peak) if ep.force_peak is not None else float("inf")
        tilt_raw = ep.metadata.get("max_tilt_deg")
        tilt = float(tilt_raw) if isinstance(tilt_raw, (int, float)) else float("inf")
        stresses.append(stress)
        tilts.append(tilt)
        if ep.success:
            successes += 1
            if stress <= COCARRY_STRESS_MAX_PROXY_N and tilt <= COCARRY_TILT_MAX_DEG:
                compliant += 1
    return CandidateScore(
        step=step,
        uri=uri,
        n_episodes=len(episodes),
        successes=successes,
        compliant_successes=compliant,
        stress_p90=float(np.percentile(np.asarray(stresses), 90)) if stresses else float("nan"),
        stress_max=float(np.max(np.asarray(stresses))) if stresses else float("nan"),
        tilt_max_deg=float(np.max(np.asarray(tilts))) if tilts else float("nan"),
    )


def select_by_rule(scores: list[CandidateScore]) -> CandidateScore:
    """Apply the preregistered rule to step-ordered scores (ADR-027 §Reporting rules).

    Maximum ``compliant_successes``; ties break to the earliest step
    (the Rung-2 precedent). Pure — unit-testable without an env.

    Raises:
        ValueError: On an empty score list.
    """
    if not scores:
        msg = "select_by_rule: no candidate scores"
        raise ValueError(msg)
    ordered = sorted(scores, key=lambda s: s.step)
    best = ordered[0]
    for row in ordered[1:]:
        if row.compliant_successes > best.compliant_successes:
            best = row
    return best


def select_checkpoint(
    *,
    candidates: list[CheckpointCandidate],
    policy_prefix: str,
    set_spec: PartnerSetSpec,
    validation_member: tuple[PartnerMemberSpec, dict[str, str]] | None,
    selection_seeds: list[int],
    episodes_per_seed: int,
    render_backend: str | None = None,
) -> CheckpointSelectionResult:
    """Evaluate every candidate on the validation dyad and apply the rule (ADR-027).

    Args:
        candidates: The training run's saved checkpoints (any order;
            scored step-ordered).
        policy_prefix: ``"happo"`` (B-AHT), ``"happo_blind"`` (B-BLIND
            — evaluated on its trained masked interface), or
            ``"joint_ego"`` (B-JOINT pair mode: the partner seat is
            the same pair checkpoint's partner-side actor and
            ``validation_member`` must be ``None``).
        set_spec: The partner set the validation member belongs to
            (custody metadata for the artifact; unused in pair mode).
        validation_member: The resolved held-out public member
            ``(member, params)`` — the prereg-named validation
            partner. ``None`` iff ``policy_prefix == "joint_ego"``.
        selection_seeds: Preregistered selection seed grid.
        episodes_per_seed: Episodes per selection seed.
        render_backend: Forwarded to the SAPIEN env factory.

    Returns:
        The :class:`CheckpointSelectionResult` (write it with
        :func:`write_selection_artifact` before touching any eval cell).

    Raises:
        ValueError: Empty candidates/seeds, or a validation-member /
            pair-mode mismatch.
    """
    from chamber.benchmarks.cocarry_eval import (
        run_cocarry_episodes_adhoc,
        run_cocarry_episodes_for_set,
    )

    if not candidates:
        msg = "select_checkpoint: no candidates"
        raise ValueError(msg)
    if not selection_seeds:
        msg = "select_checkpoint: no selection seeds"
        raise ValueError(msg)
    pair_mode = policy_prefix == "joint_ego"
    if pair_mode != (validation_member is None):
        msg = (
            "select_checkpoint: validation_member must be None iff "
            "policy_prefix='joint_ego' (B-JOINT is evaluated as the pair it "
            "trained as; ADR-011 §Decision as amended) — got "
            f"policy_prefix={policy_prefix!r}, validation_member="
            f"{'None' if validation_member is None else validation_member[0].member_name!r}."
        )
        raise ValueError(msg)

    scores: list[CandidateScore] = []
    for cand in sorted(candidates, key=lambda c: c.step):
        policy_id = f"{policy_prefix}:{cand.uri}"
        if pair_mode:
            episodes_by_seed, _ = run_cocarry_episodes_adhoc(
                policy_id=policy_id,
                partner_name="frozen_cocarry_joint",
                partner_weights=cand.uri,
                seeds=selection_seeds,
                episodes_per_seed=episodes_per_seed,
                render_backend=render_backend,
            )
        else:
            member = validation_member
            if member is None:  # pragma: no cover - guarded above
                msg = "select_checkpoint: validation_member is required"
                raise ValueError(msg)
            episodes_by_seed, _, _ = run_cocarry_episodes_for_set(
                policy_id=policy_id,
                set_spec=set_spec,
                members=[member],
                seeds=selection_seeds,
                episodes_per_seed=episodes_per_seed,
                render_backend=render_backend,
            )
        episodes = [ep for records in episodes_by_seed.values() for ep in records]
        scores.append(score_episodes(cand.step, cand.uri, episodes))

    best = select_by_rule(scores)
    return CheckpointSelectionResult(
        schema=SELECTION_SCHEMA,
        selection_rule=SELECTION_RULE,
        policy_prefix=policy_prefix,
        validation_partner=(
            "<pair>" if pair_mode else validation_member[0].member_name  # type: ignore[index]
        ),
        selection_seeds=list(selection_seeds),
        episodes_per_seed=episodes_per_seed,
        scores=scores,
        selected_step=best.step,
        selected_uri=best.uri,
    )


def write_selection_artifact(result: CheckpointSelectionResult, path: Path) -> None:
    """Write the selection-evidence JSON (ADR-027 §Reporting rules; the Rung-2 convention; I8).

    The artifact is committed next to the row's result bundle so the
    founder can re-derive the selection from the recorded scores alone.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(result)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "SELECTION_RULE",
    "SELECTION_SCHEMA",
    "CandidateScore",
    "CheckpointCandidate",
    "CheckpointSelectionResult",
    "score_episodes",
    "select_by_rule",
    "select_checkpoint",
    "write_selection_artifact",
]
