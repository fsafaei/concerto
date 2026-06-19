# SPDX-License-Identifier: Apache-2.0
r"""Rung-2 frozen-incumbent freeze manifest (ADR-026 §Decision 4; R-2026-06-B §15 Rung 2).

The freeze manifest is the load-bearing Rung-2 artifact: the auditable,
byte-frozen record that the learned co-carry incumbent was frozen against
the matched pair **before any shifted teammate was seen**, capturing
*every* reward-shaping coefficient, limit, geometry constant and bar mass
together with the checkpoint, the re-derived ``f_max``, and the matched
reference. It closes the exact "implicit-coefficient" gap that contaminated
the Stage-1 AS task: nothing the env defines may be left out of the freeze.

The completeness guarantee is enforced by construction. :func:`enumerate_cocarry_constants`
introspects :mod:`chamber.envs.cocarry` for **every** public ``COCARRY_*``
numeric (or numeric-tuple) module constant; :func:`build_manifest` copies
that full set verbatim into ``env_constants``; and
:func:`missing_constants` (driven by the Tier-1 test) re-derives the live
set and asserts the manifest covers it. Adding a new ``COCARRY_*`` reward
coefficient or limit to ``cocarry.py`` and forgetting the freeze therefore
fails the Tier-1 test — a missing-coefficient guard.

The manifest reuses existing schemas with a **new co-carry tag**
(:data:`MANIFEST_SCHEMA` = ``"cocarry_freeze_manifest/v1"``) — no bump of
``chamber.evaluation.results.SCHEMA_VERSION`` / ``chamber.comm.SCHEMA_VERSION``
/ ``chamber.evaluation.prereg.PREREG_SCHEMA_VERSION`` (invariant I9).

``f_max`` re-derivation (R-2026-06-B §15 + the structured-review follow-up):
PR #245 set ``COCARRY_STRESS_MAX_PROXY_N = 130 N`` from a 15-seed run
recorded only in a docstring (matched-success post-settle proxy p50=90,
p99=104, max=105 N). Rung 2 re-runs the matched distribution at the
pre-registered seed count (≥10-12 clusters), commits the distribution as
the artifact of record, and derives ``f_max`` as ``1.25 x matched-success
p99``. The **consistency bands are pre-stated here, before any measurement**
(no forking path):

- ``FMAX_CONSISTENCY_BAND_N`` — the re-derived ``f_max`` must land in
  ``[104, 156] N`` (±20 % of 130);
- ``MATCHED_SUCCESS_P99_BAND_N`` — the ≥12-seed matched-success stress p99
  must land in ``[85, 120] N`` (covers the #245 15-seed p50/p99/max spread).

Inside **both** ⇒ *consistent*: freeze the re-derived value (the committed
run becomes the authority; note 130 N was provisional). Outside either ⇒
*material divergence*: **STOP** and report both distributions side-by-side
with a determinism / seed-set / rig diagnosis — freeze nothing. Either way,
if ``f_max`` moves, the Rung-1 positive-control is re-scored under the new
threshold (single-arm ≈ 0, matched competence hold) — see
:func:`chamber.benchmarks.cocarry_incumbent` callers.

References:
- ADR-026 §Decision 4 (the Phase-2 forward-design ladder); §Validation
  criteria (the ≥10-12 seed-cluster forward-statistics commitment).
- ADR-002 §Decisions (the checkpoint payload+sidecar SHA-256 contract).
- ADR-009 §Decision (the frozen black-box partner the incumbent replaces).
- R-2026-06-B §15 Rung 2 (the freeze-everything-before-shift condition).
- :mod:`chamber.envs.cocarry` (the env constants frozen here).
- :func:`chamber.benchmarks.cocarry_runner.summarize` (produces the
  ``stress_p99`` / ``success_stress_p95`` the ``f_max`` derivation reads).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

#: Manifest schema tag. A NEW co-carry tag — reuses the existing
#: serialisation surface with no bump to any ``*.SCHEMA_VERSION`` (I9).
MANIFEST_SCHEMA: str = "cocarry_freeze_manifest/v1"

#: The ``f_max`` derivation rule (R-2026-06-B §15 "stress proxy"): a
#: justified high percentile of the matched-success constraint-force
#: distribution. 1.25x the matched-success p99 (the documented rule).
FMAX_P99_MULTIPLIER: float = 1.25

#: Pre-stated consistency band on the re-derived ``f_max`` (Newtons):
#: ±20 % of the provisional 130 N. Fixed BEFORE any measurement.
FMAX_CONSISTENCY_BAND_N: tuple[float, float] = (104.0, 156.0)

#: Pre-stated consistency band on the ≥12-seed matched-success stress p99
#: (Newtons): covers the PR #245 15-seed p50=90 / p99=104 / max=105 spread.
MATCHED_SUCCESS_P99_BAND_N: tuple[float, float] = (85.0, 120.0)

#: ``f_max`` classification verdicts.
FMAX_CONSISTENT: str = "consistent"
FMAX_DIVERGENT: str = "divergent"
FMAX_PENDING: str = "pending"


class CoCarryFreezeError(RuntimeError):
    """Raised when a freeze manifest is incomplete or inconsistent (ADR-026 §Decision 4)."""


def _is_frozen_grade_constant(value: object) -> bool:
    """Whether a module value is a frozen-grade co-carry constant (ADR-026 §Decision 4).

    Captures scalars (``int`` / ``float``, excluding ``bool``) and
    numeric tuples/lists (e.g. the goal-centroid geometry). These are the
    pre-registration-grade constants the manifest must freeze; non-numeric
    module objects (strings, arrays, NamedTuples, functions) are skipped.

    Scope caveat: a future **non-numeric** ``COCARRY_*`` constant (a string
    or dict — e.g. a named reward mode) would be silently excluded here and
    so escape the manifest-completeness guard. If such a constant is added,
    broaden this predicate (and the manifest serialisation) to cover it.
    """
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, (tuple, list)) and value:
        return all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in value)
    return False


def _jsonable(value: object) -> Any:  # noqa: ANN401 - scalar or numeric tuple
    """Coerce a frozen-grade constant to a JSON-serialisable form (tuples → lists)."""
    if isinstance(value, (tuple, list)):
        return [float(x) if isinstance(x, float) else int(x) for x in value]
    return value


def enumerate_cocarry_constants() -> dict[str, Any]:
    """Every public ``COCARRY_*`` numeric constant in the co-carry env (ADR-026 §Decision 4).

    The single source of truth for the freeze's completeness guarantee:
    introspects the live module so a newly-added reward coefficient or
    limit is captured automatically (and a forgotten freeze fails the
    Tier-1 completeness test). Tier-1-safe (the cocarry module's top level
    imports only numpy + the seeding harness).

    Returns:
        ``{constant_name: jsonable_value}`` for every public ``COCARRY_*``
        scalar or numeric tuple.
    """
    from chamber.envs import cocarry

    out: dict[str, Any] = {}
    for name, value in vars(cocarry).items():
        if name.startswith("COCARRY_") and _is_frozen_grade_constant(value):
            out[name] = _jsonable(value)
    return out


def derive_fmax_from_p99(matched_success_stress_p99: float) -> float:
    """``f_max`` = multiplier x matched-success p99 (ADR-026 §Decision 4; R-2026-06-B §15)."""
    return float(FMAX_P99_MULTIPLIER) * float(matched_success_stress_p99)


def classify_fmax(*, fmax_value: float, matched_success_stress_p99: float) -> str:
    """Verdict against the pre-stated bands (ADR-026 §Decision 4; R-2026-06-B §15).

    Returns :data:`FMAX_CONSISTENT` iff the re-derived ``f_max`` is within
    :data:`FMAX_CONSISTENCY_BAND_N` **and** the matched-success p99 is
    within :data:`MATCHED_SUCCESS_P99_BAND_N`; else :data:`FMAX_DIVERGENT`.
    Material divergence is a STOP-and-report condition, not a pick-a-number.
    """
    flo, fhi = FMAX_CONSISTENCY_BAND_N
    plo, phi = MATCHED_SUCCESS_P99_BAND_N
    fmax_ok = flo <= float(fmax_value) <= fhi
    p99_ok = plo <= float(matched_success_stress_p99) <= phi
    return FMAX_CONSISTENT if (fmax_ok and p99_ok) else FMAX_DIVERGENT


def _sha256_of_text(text: str) -> str:
    """SHA-256 hex digest of a UTF-8 string (content hash for config / module)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_of_file(path: Path) -> str:
    """SHA-256 hex digest of a file's bytes (content hash; ADR-026 §Decision 4)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class FMaxRecord:
    """The frozen ``f_max`` block (ADR-026 §Decision 4; R-2026-06-B §15).

    Attributes:
        value_n: The frozen over-stress ceiling, Newtons.
        derivation: Human-readable derivation rule.
        multiplier: :data:`FMAX_P99_MULTIPLIER`.
        matched_success_stress_p99_n: The ≥12-seed matched-success p99 the
            value was derived from.
        consistency_band_n: Pre-stated band on ``value_n``.
        matched_success_p99_band_n: Pre-stated band on the p99.
        distribution_artifact: Repo-relative path to the committed
            constraint-force distribution (the artifact of record).
        status: :data:`FMAX_CONSISTENT` / :data:`FMAX_DIVERGENT` /
            :data:`FMAX_PENDING`.
        previous_provisional_n: The pre-#245 provisional value (130 N),
            recorded so the audit trail shows what was superseded.
    """

    value_n: float
    matched_success_stress_p99_n: float
    status: str
    distribution_artifact: str | None = None
    multiplier: float = FMAX_P99_MULTIPLIER
    derivation: str = "1.25 x matched-success stress p99 (R-2026-06-B §15)"
    consistency_band_n: tuple[float, float] = FMAX_CONSISTENCY_BAND_N
    matched_success_p99_band_n: tuple[float, float] = MATCHED_SUCCESS_P99_BAND_N
    previous_provisional_n: float = 130.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain JSON-able dict (ADR-026 §Decision 4)."""
        return {
            "value_n": float(self.value_n),
            "matched_success_stress_p99_n": float(self.matched_success_stress_p99_n),
            "status": self.status,
            "distribution_artifact": self.distribution_artifact,
            "multiplier": float(self.multiplier),
            "derivation": self.derivation,
            "consistency_band_n": list(self.consistency_band_n),
            "matched_success_p99_band_n": list(self.matched_success_p99_band_n),
            "previous_provisional_n": float(self.previous_provisional_n),
        }


@dataclass(frozen=True)
class MatchedReference:
    """The frozen Rung-1 matched-pair reference the incumbent must reach (ADR-026 §Validation).

    Attributes:
        success_rate: Matched-pair joint-success rate (the training target).
        n_seed_clusters: Seed-cluster count the rate was measured over
            (≥10-12 per ADR-026 §Validation criteria; 5-6 is exploration
            only).
        artifact: Repo-relative path to the committed measurement.
    """

    success_rate: float
    n_seed_clusters: int
    artifact: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain JSON-able dict (ADR-026 §Validation criteria)."""
        return {
            "success_rate": float(self.success_rate),
            "n_seed_clusters": int(self.n_seed_clusters),
            "artifact": self.artifact,
        }


@dataclass(frozen=True)
class CheckpointRecord:
    """The frozen incumbent checkpoint pointer (ADR-002 §Decisions).

    Attributes:
        uri: ``local://...`` checkpoint URI.
        sha256: SHA-256 of the ``.pt`` payload (the sidecar digest).
        seed: Training seed that produced this incumbent.
        step: Training-frame step of the snapshot.
    """

    uri: str
    sha256: str
    seed: int
    step: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain JSON-able dict (ADR-002 §Decisions)."""
        return {
            "uri": self.uri,
            "sha256": self.sha256,
            "seed": int(self.seed),
            "step": None if self.step is None else int(self.step),
        }


@dataclass(frozen=True)
class FreezeManifest:
    """The complete Rung-2 freeze manifest (ADR-026 §Decision 4; R-2026-06-B §15).

    Captures the full frozen state of the learned co-carry incumbent: every
    ``COCARRY_*`` constant, the re-derived ``f_max``, the matched reference,
    the env-module + training-config content hashes, and the checkpoint.
    """

    schema: str
    rung: int
    env_constants: dict[str, Any]
    fmax: FMaxRecord
    matched_reference: MatchedReference
    training_config_sha256: str
    env_module_sha256: str
    checkpoint: CheckpointRecord | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the whole manifest to a plain JSON-able dict (ADR-026 §Decision 4)."""
        return {
            "schema": self.schema,
            "rung": self.rung,
            "env_constants": dict(self.env_constants),
            "fmax": self.fmax.to_dict(),
            "matched_reference": self.matched_reference.to_dict(),
            "training_config_sha256": self.training_config_sha256,
            "env_module_sha256": self.env_module_sha256,
            "checkpoint": None if self.checkpoint is None else self.checkpoint.to_dict(),
            "notes": list(self.notes),
        }


def build_manifest(
    *,
    matched_reference_success_rate: float,
    n_seed_clusters: int,
    matched_success_stress_p99_n: float,
    fmax_value_n: float | None = None,
    distribution_artifact: str | None = None,
    matched_reference_artifact: str | None = None,
    config_path: Path | str | None = None,
    env_module_path: Path | str | None = None,
    checkpoint: CheckpointRecord | None = None,
    notes: list[str] | None = None,
) -> FreezeManifest:
    """Assemble a complete Rung-2 freeze manifest (ADR-026 §Decision 4; R-2026-06-B §15).

    ``env_constants`` is populated from :func:`enumerate_cocarry_constants`
    (the completeness guarantee). ``f_max`` is the supplied ``fmax_value_n``
    (or, when ``None``, re-derived as ``1.25 x matched_success_stress_p99_n``)
    and is classified against the pre-stated bands. The content hashes
    default to the canonical paths when not given.

    Args:
        matched_reference_success_rate: Matched-pair joint-success rate
            (the training target).
        n_seed_clusters: Seed-cluster count the reference was measured over.
        matched_success_stress_p99_n: The matched-success stress p99 (N).
        fmax_value_n: The frozen ``f_max``; ``None`` re-derives from p99.
        distribution_artifact: Repo-relative path to the committed
            constraint-force distribution (artifact of record).
        matched_reference_artifact: Repo-relative path to the committed
            matched-reference measurement.
        config_path: Path to ``cocarry_matched.yaml`` (content-hashed).
        env_module_path: Path to ``chamber/envs/cocarry.py`` (content-hashed).
        checkpoint: The frozen-incumbent :class:`CheckpointRecord` (``None``
            for a pre-checkpoint completeness build).
        notes: Free-form audit notes.

    Returns:
        The assembled :class:`FreezeManifest`.
    """
    fmax_value = (
        float(fmax_value_n)
        if fmax_value_n is not None
        else derive_fmax_from_p99(matched_success_stress_p99_n)
    )
    status = classify_fmax(
        fmax_value=fmax_value, matched_success_stress_p99=matched_success_stress_p99_n
    )
    fmax = FMaxRecord(
        value_n=fmax_value,
        matched_success_stress_p99_n=float(matched_success_stress_p99_n),
        status=status,
        distribution_artifact=distribution_artifact,
    )
    matched = MatchedReference(
        success_rate=float(matched_reference_success_rate),
        n_seed_clusters=int(n_seed_clusters),
        artifact=matched_reference_artifact,
    )
    config_sha = (
        sha256_of_file(Path(config_path)) if config_path is not None else _sha256_of_text("")
    )
    env_sha = (
        sha256_of_file(Path(env_module_path))
        if env_module_path is not None
        else _sha256_of_text("")
    )
    return FreezeManifest(
        schema=MANIFEST_SCHEMA,
        rung=2,
        env_constants=enumerate_cocarry_constants(),
        fmax=fmax,
        matched_reference=matched,
        training_config_sha256=config_sha,
        env_module_sha256=env_sha,
        checkpoint=checkpoint,
        notes=list(notes or []),
    )


def missing_constants(manifest: Mapping[str, Any] | FreezeManifest) -> list[str]:
    """Frozen-grade ``COCARRY_*`` constants absent from the manifest (ADR-026 §Decision 4).

    The completeness guard the Tier-1 test asserts is empty: re-derives the
    live set via :func:`enumerate_cocarry_constants` and returns any name
    not present in the manifest's ``env_constants``. A non-empty result
    means the freeze is incomplete — a partially-frozen incumbent
    reintroduces the Stage-1 AS defect (R-2026-06-B §15).

    Args:
        manifest: A :class:`FreezeManifest` or its serialised dict.

    Returns:
        Sorted list of missing constant names (empty ⇒ complete).
    """
    env_constants = (
        manifest.env_constants
        if isinstance(manifest, FreezeManifest)
        else dict(manifest.get("env_constants", {}))
    )
    live = enumerate_cocarry_constants()
    return sorted(name for name in live if name not in env_constants)


def assert_manifest_complete(manifest: Mapping[str, Any] | FreezeManifest) -> None:
    """Raise :class:`CoCarryFreezeError` if any frozen-grade constant is missing (ADR-026 §D4)."""
    missing = missing_constants(manifest)
    if missing:
        raise CoCarryFreezeError(
            "Rung-2 freeze manifest is incomplete — missing frozen-grade co-carry "
            f"constants: {missing}. A partially-frozen incumbent reintroduces the "
            "Stage-1 AS implicit-coefficient defect (R-2026-06-B §15; ADR-026 "
            "§Decision 4). Add them to env_constants before freezing."
        )


def write_manifest(manifest: FreezeManifest, path: Path | str) -> Path:
    """Write the manifest to JSON (ADR-026 §Decision 4; no schema bump — I9).

    Raises:
        CoCarryFreezeError: If the manifest is incomplete (the write is
            refused so an incomplete freeze never lands on disk).
    """
    assert_manifest_complete(manifest)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest.to_dict(), sort_keys=True, indent=2), encoding="utf-8")
    return out


def load_manifest(path: Path | str) -> dict[str, Any]:
    """Load a freeze manifest JSON into a plain dict (ADR-026 §Decision 4)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "FMAX_CONSISTENCY_BAND_N",
    "FMAX_CONSISTENT",
    "FMAX_DIVERGENT",
    "FMAX_P99_MULTIPLIER",
    "FMAX_PENDING",
    "MANIFEST_SCHEMA",
    "MATCHED_SUCCESS_P99_BAND_N",
    "CheckpointRecord",
    "CoCarryFreezeError",
    "FMaxRecord",
    "FreezeManifest",
    "MatchedReference",
    "assert_manifest_complete",
    "build_manifest",
    "classify_fmax",
    "derive_fmax_from_p99",
    "enumerate_cocarry_constants",
    "load_manifest",
    "missing_constants",
    "sha256_of_file",
    "write_manifest",
]
