# SPDX-License-Identifier: Apache-2.0
"""Partner-set mechanics — versioned per-task sets (ADR-009 §Decision as amended 2026-07-05).

The ADR-009 v1.0 right-sizing amendment scopes the zoo **per admitted
task**: each admitted task (and each confirmed Tier-1 control) carries a
versioned partner set of named members across two strata — a
parameterized scripted family now, frozen learned checkpoints joining as
by-products of baseline training (a set-version bump per ADR-027
§Versioning; old result bundles keep referencing the version they ran
against). This module is the set machinery:

- :class:`PartnerMemberSpec` / :class:`PartnerSetSpec` — frozen,
  identity-bearing specs. A member's behavioural parameters are folded
  into its :class:`~chamber.partners.api.PartnerSpec` identity hash via
  the ``member://`` ``weights_uri`` (see :func:`member_weights_uri`), so
  two parameterizations of one registry class carry distinct
  ``partner_id`` hashes — the ADR-006 swap-detection contract keeps
  working across set members without touching the hash mechanism.
- The deterministic 70/30 public/private split
  (:func:`compute_split`): order members by ``partner_id`` hash, assign
  the first ``⌈0.7·N⌉`` to public — no hand-picking. Registration
  recomputes the rule and loud-fails on any committed drift.
- The withheld-parameters scheme for private members (the I4-adjacent
  discipline: published hashes, withheld parameters): every member's
  exact parameters are drawn from its **committed parameter box** by
  :func:`derive_member_params` under a **maintainer-held root seed**
  (:data:`PRIVATE_PARAMS_ENV`; never committed). Public members' drawn
  values are committed as literals next to their digest; private
  members commit only the box + the SHA-256 digest of the withheld
  values, so the exact behavioural parameters are not inferable from
  public code while any locally-derived reconstruction is
  digest-verified before use (ADR-018 §Decision custody style).
- A per-``set_id`` version registry mirroring :mod:`chamber.tasks`
  (ADR-027 §Versioning; ADR-009 §Decision registry error style).

Determinism: every draw routes through
:func:`concerto.training.seeding.derive_substream` (ADR-002 P6).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from chamber.partners.api import PartnerSpec
from concerto.training.seeding import derive_substream

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Mapping, Sequence

#: Fraction of a set's members assigned to the public split (ADR-009
#: §Decision as amended 2026-07-05: "the 70/30 public/private split is
#: realized at partner-set level"). Applied as ``⌈0.7·N⌉`` public.
PUBLIC_SPLIT_FRACTION: float = 0.7

#: Environment variable carrying the maintainer-held private-parameter
#: root seed (ADR-009 §Decision as amended; the I4-adjacent discipline).
#: A large random integer held by the maintainer, never committed;
#: private members' exact parameters derive from it via
#: :func:`derive_member_params` and verify against their committed
#: SHA-256 digest before any use.
PRIVATE_PARAMS_ENV: str = "CHAMBER_PRIVATE_PARTNER_SEED"

#: ``derive_substream`` label pattern for member-parameter derivation
#: (ADR-002 P6). Keyed on ``set_id`` + ``member_name`` only — NOT the
#: set version — so a member's parameters (and hence its
#: ``partner_id``) are stable across set-version bumps (ADR-027
#: §Versioning: adding members produces ``version+1`` without
#: disturbing existing member identities).
MEMBER_PARAMS_SUBSTREAM: str = "chamber.partners.sets.{set_id}.{member_name}"

#: Decimal places float parameter draws are rounded to before being
#: stringified (keeps the committed literals short and the digest
#: material canonical).
PARAM_FLOAT_DECIMALS: int = 4

#: The split vocabulary (ADR-009 §Decision as amended 2026-07-05).
SplitLabel = Literal["public", "private"]


class WithheldParametersError(RuntimeError):
    """Private-member parameters requested but not available/verified (ADR-009 as amended).

    Raised when the maintainer-held seed (:data:`PRIVATE_PARAMS_ENV`) is
    absent or malformed, or when a locally-derived reconstruction fails
    its committed SHA-256 digest check (wrong seed — never silently
    build a behaviourally-wrong private member).
    """


def canonical_params_json(params: Mapping[str, str]) -> str:
    """Canonical JSON encoding of a member's behavioural parameters (ADR-009 as amended).

    Sorted keys, no whitespace — the digest material for
    :func:`params_sha256`, so two processes (and the committed literal
    vs a seed-derived reconstruction) hash identically.
    """
    return json.dumps({str(k): str(v) for k, v in params.items()}, sort_keys=True)


def params_sha256(params: Mapping[str, str]) -> str:
    """SHA-256 hex digest of the canonical parameter encoding (ADR-009 as amended; ADR-018).

    The committed custody hash for a member's behavioural parameters:
    published for every member, verifying public literals and private
    seed-derived reconstructions alike (published hashes, withheld
    parameters — the I4-adjacent discipline).
    """
    return hashlib.sha256(canonical_params_json(params).encode("utf-8")).hexdigest()


def member_weights_uri(registry_class: str, digest: str) -> str:
    """The ``member://`` URI folding behavioural parameters into identity (ADR-009 as amended).

    :attr:`chamber.partners.api.PartnerSpec.partner_id` deliberately
    hashes ``(class_name, seed, checkpoint_step, weights_uri)`` and not
    ``extra`` — so two scripted-family members that differ only in their
    construction parameters would collide. Set members therefore carry
    ``weights_uri = "member://<registry_class>/<params_sha256>"``: the
    parameters' digest rides in an identity-bearing field, distinct
    parameterizations get distinct ``partner_id`` hashes, and the
    ADR-006 risk #3 swap-detection contract (conformal ``lambda``
    re-init on partner change) extends to set members with the hash
    mechanism untouched. Scripted partners never dereference
    ``weights_uri``, so the URI is inert at construction time.
    """
    return f"member://{registry_class}/{digest}"


class ParamRange(BaseModel):
    """One committed parameter box edge for a set member (ADR-009 §Decision as amended).

    The box is public for every member (it is the capability-matching
    statement); the exact drawn value is committed for public members
    and withheld for private ones. ``lo == hi`` pins a parameter to a
    preregistered point value (e.g. the handover Gate-0 presenter
    numbers, tag ``prereg-handover-place-gate0-rev2-2026-06-26``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    lo: float
    hi: float
    kind: Literal["float", "int"] = "float"

    @model_validator(mode="after")
    def _check_bounds(self) -> ParamRange:
        """Loud-fail on an inverted box (ADR-009 as amended)."""
        if self.hi < self.lo:
            msg = f"ParamRange requires lo <= hi; got [{self.lo}, {self.hi}]"
            raise ValueError(msg)
        if self.kind == "int" and (self.lo != int(self.lo) or self.hi != int(self.hi)):
            msg = f"int ParamRange requires integer bounds; got [{self.lo}, {self.hi}]"
            raise ValueError(msg)
        return self

    def contains(self, raw: str) -> bool:
        """Whether a stringified drawn value lies inside the box (ADR-009 as amended)."""
        try:
            value = float(raw)
        except ValueError:
            return False
        if self.kind == "int" and value != int(value):
            return False
        return self.lo - 1e-12 <= value <= self.hi + 1e-12


def derive_member_params(
    set_id: str,
    member_name: str,
    box: Mapping[str, ParamRange],
    *,
    root_seed: int,
) -> dict[str, str]:
    """Draw a member's exact parameters from its committed box (ADR-009 as amended; ADR-002 P6).

    Deterministic in ``(set_id, member_name, box, root_seed)``: one
    ``derive_substream`` per member (:data:`MEMBER_PARAMS_SUBSTREAM`),
    draws taken per parameter in sorted-key order — so the maintainer
    (holding the private root seed) re-derives byte-identical values in
    any process, and the committed digest verifies the reconstruction.
    Floats are drawn uniform on ``[lo, hi]`` and rounded to
    :data:`PARAM_FLOAT_DECIMALS`; ints uniform on the inclusive range.
    """
    label = MEMBER_PARAMS_SUBSTREAM.format(set_id=set_id, member_name=member_name)
    rng = derive_substream(label, root_seed=root_seed).default_rng()
    params: dict[str, str] = {}
    for key in sorted(box):
        edge = box[key]
        if edge.kind == "int":
            params[key] = str(int(rng.integers(int(edge.lo), int(edge.hi) + 1)))
        else:
            value = round(float(rng.uniform(edge.lo, edge.hi)), PARAM_FLOAT_DECIMALS)
            params[key] = repr(value)
    return params


def compose_member_extra(registry_class: str, params: Mapping[str, str]) -> dict[str, str]:
    """Map scalar behavioural parameters onto a class's ``spec.extra`` keys (ADR-009 as amended).

    Parameter boxes are scalar-per-key; most partner classes read
    scalar ``extra`` keys verbatim. The one compound key today:
    :class:`chamber.partners.heuristic.ScriptedHeuristicPartner` reads
    ``target_xy`` as an ``"x,y"`` string, composed here from the
    ``target_x`` / ``target_y`` scalars so the box machinery stays
    uniform.
    """
    extra = {str(k): str(v) for k, v in params.items()}
    if registry_class == "scripted_heuristic" and "target_x" in extra and "target_y" in extra:
        extra["target_xy"] = f"{extra.pop('target_x')},{extra.pop('target_y')}"
    return extra


class PartnerMemberSpec(BaseModel):
    """One named member of a versioned partner set (ADR-009 §Decision as amended 2026-07-05).

    Attributes:
        member_name: Set-unique name (card file name, roster key,
            ``partners.json`` member key).
        registry_class: The :func:`chamber.partners.registry.register_partner`
            key the member is built from (the scripted stratum
            parameterizes one class into many members).
        role: Seat the member plays where the task is asymmetric
            (``partner_arm`` / ``presenter`` / ``partner``).
        split: Committed split assignment — validated against the
            recomputed :func:`compute_split` rule at set registration
            (no hand-picking).
        seed: :class:`~chamber.partners.api.PartnerSpec` seed identity
            field (``0`` for scripted members; CB-06 learned members
            carry their training seed).
        checkpoint_step: Checkpoint identity field (``None`` scripted).
        param_box: Committed per-parameter draw boxes (public for every
            member — the capability-matching statement).
        params: Exact behavioural parameters — committed literals for
            public members, ``None`` (withheld) for private members.
        params_sha256: SHA-256 custody digest of the exact parameters
            (:func:`params_sha256`) — committed for every member; the
            verification anchor for private reconstructions (I4-adjacent
            discipline; ADR-018 §Decision custody style).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    member_name: str
    registry_class: str
    role: str
    split: SplitLabel
    seed: int = 0
    checkpoint_step: int | None = None
    param_box: dict[str, ParamRange]
    params: dict[str, str] | None
    params_sha256: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def _check_params(self) -> PartnerMemberSpec:
        """Enforce the split/withholding invariants (ADR-009 as amended; ADR-018)."""
        if self.split == "private":
            if self.params is not None:
                msg = (
                    f"member {self.member_name!r} is private but carries committed "
                    "parameter values — private parameters are withheld (I4 discipline)"
                )
                raise ValueError(msg)
            return self
        if self.params is None:
            msg = f"member {self.member_name!r} is public but carries no parameter values"
            raise ValueError(msg)
        if set(self.params) != set(self.param_box):
            msg = (
                f"member {self.member_name!r}: params keys {sorted(self.params)} != "
                f"param_box keys {sorted(self.param_box)}"
            )
            raise ValueError(msg)
        for key, raw in self.params.items():
            if not self.param_box[key].contains(raw):
                edge = self.param_box[key]
                msg = (
                    f"member {self.member_name!r}: params[{key!r}] = {raw!r} outside "
                    f"the committed box [{edge.lo}, {edge.hi}]"
                )
                raise ValueError(msg)
        digest = params_sha256(self.params)
        if digest != self.params_sha256:
            msg = (
                f"member {self.member_name!r}: committed params_sha256 "
                f"{self.params_sha256} != recomputed {digest}"
            )
            raise ValueError(msg)
        return self

    @property
    def weights_uri(self) -> str:
        """The identity-bearing ``member://`` URI (ADR-009 as amended)."""
        return member_weights_uri(self.registry_class, self.params_sha256)

    @property
    def partner_id(self) -> str:
        """The member's :attr:`~chamber.partners.api.PartnerSpec.partner_id` (ADR-006 risk #3).

        Computable for private members without their withheld
        parameters — identity rides on the committed digest, so the
        split rule, cards, and bundle custody hashes never need the
        parameter values themselves.
        """
        return self._identity_spec().partner_id

    def _identity_spec(self, extra: dict[str, str] | None = None) -> PartnerSpec:
        return PartnerSpec(
            class_name=self.registry_class,
            seed=self.seed,
            checkpoint_step=self.checkpoint_step,
            weights_uri=self.weights_uri,
            extra=extra or {},
        )

    def partner_spec(
        self,
        *,
        params: Mapping[str, str],
        seat_extra: Mapping[str, str] | None = None,
    ) -> PartnerSpec:
        """Build the loadable :class:`~chamber.partners.api.PartnerSpec` (ADR-009 §Decision).

        Args:
            params: The member's exact behavioural parameters — the
                committed literals for public members, or a
                digest-verified private reconstruction from
                :func:`resolve_member_params`. Verified against
                :attr:`params_sha256` here as well (defence in depth).
            seat_extra: Task-seat metadata (uid, base pose, action_dim
                …) supplied by the runner; identity-inert (``extra`` is
                excluded from the hash by design, plan/04 §3.1).

        Raises:
            WithheldParametersError: On a digest mismatch.
        """
        digest = params_sha256(params)
        if digest != self.params_sha256:
            msg = (
                f"member {self.member_name!r}: supplied parameters hash {digest}, "
                f"committed digest {self.params_sha256} — refusing to build a "
                "behaviourally-unverified member (ADR-018 custody)"
            )
            raise WithheldParametersError(msg)
        extra = dict(seat_extra or {})
        extra.update(compose_member_extra(self.registry_class, params))
        return self._identity_spec(extra)


def public_member_count(n_members: int) -> int:
    """``⌈0.7·N⌉`` — the public-slot count of an N-member set (ADR-009 as amended)."""
    return math.ceil(PUBLIC_SPLIT_FRACTION * n_members)


def compute_split(partner_ids: Sequence[str]) -> dict[str, SplitLabel]:
    """The deterministic public/private split rule (ADR-009 §Decision as amended 2026-07-05).

    Order members by ``partner_id`` hash (ascending lexicographic) and
    assign the first ``⌈0.7·N⌉`` to public — no hand-picking. Pure
    function of the committed hashes, so anyone can recompute the split
    from a set's published roster; registration enforces agreement.

    Raises:
        ValueError: On duplicate ``partner_id`` hashes (two members may
            not share an identity).
    """
    ids = list(partner_ids)
    if len(set(ids)) != len(ids):
        msg = "compute_split requires distinct partner_id hashes"
        raise ValueError(msg)
    n_public = public_member_count(len(ids))
    ordered = sorted(ids)
    return {pid: ("public" if i < n_public else "private") for i, pid in enumerate(ordered)}


class PartnerSetSpec(BaseModel):
    """A versioned per-task partner set (ADR-009 §Decision as amended; ADR-027 §Versioning).

    Attributes:
        set_id: Stable set identity (``<task_id>_partners``).
        version: Set version; **any** membership or parameter change
            bumps it (ADR-027 §Versioning — CB-06's frozen learned
            members join as ``version+1``; old bundles keep referencing
            the version they ran against).
        task_id: The ADR-027 task the set serves.
        task_version: The task version the set was built for.
        members: The roster. Split assignments are validated against
            :func:`compute_split` at construction.
        floor: Committed matched-pair success floor every member must
            clear with the reference ego on the floor probe before
            admission to the set (a member no ego can work with
            measures nothing — the co-insert Gate-0 lesson: a member
            that freezes is a wall, not a partner).
        floor_probe: Which probe the floor is evaluated on.
            ``"fingerprint"`` — the fingerprint probe itself (co-carry,
            pick-place). ``"free_regrasp"`` — the same probe with the
            re-grasp budget removed (handover-place: the canonical-cell
            probe *intentionally* degrades mismatch members — that gap
            is the measured coupling, not incompetence — so competence
            is judged at the Gate-0 free-re-grasp endpoint where budget
            pressure is off and only a wall scores low).
        probe_seeds: Committed probe-suite cluster seeds (ADR-002 P6).
        probe_episodes_per_seed: Episodes per probe seed (~20 total).
        notes: Free-text provenance.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    set_id: str
    version: int = Field(ge=1)
    task_id: str
    task_version: int
    members: list[PartnerMemberSpec] = Field(min_length=1)
    floor: float = Field(ge=0.0, le=1.0)
    floor_probe: Literal["fingerprint", "free_regrasp"] = "fingerprint"
    probe_seeds: list[int] = Field(min_length=1)
    probe_episodes_per_seed: int = Field(ge=1)
    notes: str = ""

    @model_validator(mode="after")
    def _check_roster(self) -> PartnerSetSpec:
        """Recompute the split rule and reject drift (ADR-009 as amended: no hand-picking)."""
        names = [m.member_name for m in self.members]
        if len(set(names)) != len(names):
            msg = f"set {self.set_id!r}: duplicate member_name in roster"
            raise ValueError(msg)
        expected = compute_split([m.partner_id for m in self.members])
        for member in self.members:
            if member.split != expected[member.partner_id]:
                msg = (
                    f"set {self.set_id!r}: member {member.member_name!r} committed as "
                    f"{member.split!r} but the deterministic split rule assigns "
                    f"{expected[member.partner_id]!r} (ADR-009 as amended: order by "
                    "partner_id, first ⌈0.7·N⌉ public — no hand-picking)"
                )
                raise ValueError(msg)
        return self

    @property
    def slug(self) -> str:
        """``set_id@vN`` — the versioned set handle (ADR-027 §Versioning)."""
        return f"{self.set_id}@v{self.version}"

    @property
    def public_members(self) -> list[PartnerMemberSpec]:
        """Roster order, public split only (ADR-009 as amended)."""
        return [m for m in self.members if m.split == "public"]

    @property
    def private_members(self) -> list[PartnerMemberSpec]:
        """Roster order, private split only (ADR-009 as amended)."""
        return [m for m in self.members if m.split == "private"]


# ---------------------------------------------------------------------------
# Withheld-parameter resolution (the maintainer-held-seed scheme).
# ---------------------------------------------------------------------------


def private_seed_available() -> bool:
    """Whether the maintainer-held private-parameter seed is present (ADR-009 as amended)."""
    raw = os.environ.get(PRIVATE_PARAMS_ENV)
    if raw is None:
        return False
    try:
        int(raw)
    except ValueError:
        return False
    return True


def _private_root_seed() -> int:
    raw = os.environ.get(PRIVATE_PARAMS_ENV)
    if raw is None:
        msg = (
            f"private-member parameters are withheld (ADR-009 as amended: published "
            f"hashes, withheld parameters); set {PRIVATE_PARAMS_ENV} to the "
            "maintainer-held root seed to derive them locally"
        )
        raise WithheldParametersError(msg)
    try:
        return int(raw)
    except ValueError as exc:
        msg = f"{PRIVATE_PARAMS_ENV} must be an integer root seed; got {raw!r}"
        raise WithheldParametersError(msg) from exc


def resolve_member_params(set_spec: PartnerSetSpec, member: PartnerMemberSpec) -> dict[str, str]:
    """Return a member's exact parameters, digest-verified (ADR-009 as amended; ADR-018).

    Public members return their committed literals. Private members
    require the maintainer-held seed (:data:`PRIVATE_PARAMS_ENV`): the
    parameters are re-derived from the committed box via
    :func:`derive_member_params` and verified against the committed
    digest — a wrong seed loud-fails rather than silently building a
    behaviourally-wrong member.

    Raises:
        WithheldParametersError: Seed absent/malformed, or derived
            parameters fail the committed digest.
    """
    if member.params is not None:
        return dict(member.params)
    derived = derive_member_params(
        set_spec.set_id, member.member_name, member.param_box, root_seed=_private_root_seed()
    )
    digest = params_sha256(derived)
    if digest != member.params_sha256:
        msg = (
            f"member {member.member_name!r}: parameters derived from "
            f"{PRIVATE_PARAMS_ENV} hash {digest}, committed digest "
            f"{member.params_sha256} — wrong maintainer seed (ADR-018 custody)"
        )
        raise WithheldParametersError(msg)
    return derived


def resolve_set_members(
    set_spec: PartnerSetSpec,
    *,
    include_private: bool = False,
) -> list[tuple[PartnerMemberSpec, dict[str, str]]]:
    """Resolve the runnable roster with exact parameters (ADR-009 as amended).

    Public members only by default (the shipped evaluation surface);
    ``include_private=True`` adds private members and therefore
    requires the withheld parameters to be derivable locally
    (:func:`resolve_member_params`).

    Returns:
        ``(member, params)`` pairs in roster order.

    Raises:
        WithheldParametersError: ``include_private`` without the
            maintainer-held seed, or on any digest mismatch.
    """
    selected = set_spec.members if include_private else set_spec.public_members
    return [(member, resolve_member_params(set_spec, member)) for member in selected]


# ---------------------------------------------------------------------------
# Set registry (mirrors chamber.tasks.registry; ADR-027 §Versioning).
# ---------------------------------------------------------------------------

# set_id -> {version -> PartnerSetSpec}
_SET_REGISTRY: dict[str, dict[int, PartnerSetSpec]] = {}


def register_partner_set(
    builder: Callable[[], PartnerSetSpec],
) -> Callable[[], PartnerSetSpec]:
    """Register the set returned by ``builder`` (ADR-027 §Versioning; ADR-009 registry style).

    Decorator over a zero-argument builder; the spec is built (running
    the split-rule validation) and registered eagerly. Re-registering an
    existing ``set_id@vN`` raises ``ValueError`` — a changed set must
    bump its version, never overwrite (ADR-027 §Versioning).
    """
    spec = builder()
    versions = _SET_REGISTRY.setdefault(spec.set_id, {})
    if spec.version in versions:
        msg = f"partner set {spec.slug!r} is already registered"
        raise ValueError(msg)
    versions[spec.version] = spec
    return builder


def get_partner_set(set_id: str, version: int | None = None) -> PartnerSetSpec:
    """Look up a registered set (ADR-027 §Versioning; ADR-009 §Decision error style).

    ``version=None`` resolves to the latest registered version. Unknown
    ids/versions raise ``KeyError`` listing the known keys.
    """
    versions = _SET_REGISTRY.get(set_id)
    if versions is None:
        known = ", ".join(sorted(_SET_REGISTRY)) or "<none>"
        msg = f"unknown partner-set id {set_id!r}; registered set ids: {known}"
        raise KeyError(msg)
    if version is None:
        version = max(versions)
    if version not in versions:
        known_versions = ", ".join(str(v) for v in sorted(versions))
        msg = (
            f"unknown version {version} for partner set {set_id!r}; "
            f"registered versions: {known_versions}"
        )
        raise KeyError(msg)
    return versions[version]


def list_partner_sets() -> list[str]:
    """Sorted ``set_id@vN`` slugs of every registered set version (ADR-027 §Versioning)."""
    return sorted(spec.slug for versions in _SET_REGISTRY.values() for spec in versions.values())


def parse_set_slug(raw: str) -> tuple[str, int | None]:
    """Split ``set_id[@vN]`` into ``(set_id, version | None)`` (ADR-027 §Versioning)."""
    set_id, sep, version = raw.partition("@v")
    return (set_id, int(version)) if sep else (set_id, None)


__all__ = [
    "MEMBER_PARAMS_SUBSTREAM",
    "PARAM_FLOAT_DECIMALS",
    "PRIVATE_PARAMS_ENV",
    "PUBLIC_SPLIT_FRACTION",
    "ParamRange",
    "PartnerMemberSpec",
    "PartnerSetSpec",
    "SplitLabel",
    "WithheldParametersError",
    "canonical_params_json",
    "compose_member_extra",
    "compute_split",
    "derive_member_params",
    "get_partner_set",
    "list_partner_sets",
    "member_weights_uri",
    "params_sha256",
    "parse_set_slug",
    "private_seed_available",
    "public_member_count",
    "register_partner_set",
    "resolve_member_params",
    "resolve_set_members",
]
