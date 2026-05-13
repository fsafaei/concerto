# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false
#
# Same rationale as :mod:`chamber.benchmarks.ego_ppo_trainer`: torch's
# stub files do not export ``tanh`` / ``from_numpy`` via ``__all__``
# even though they are public API per the official docs. Suppress
# file-locally so per-line ``type: ignore`` clutter stays out.
"""Frozen MAPPO partner — Stage-3 draft-zoo entry #2 (ADR-009 §Decision; plan/04 §3.5).

The wrapper loads a torch state-dict via
:func:`concerto.training.checkpoints.load_checkpoint` (which verifies the
SHA-256 sidecar), reconstructs the actor MLP from the inferred layer
shapes, freezes every parameter (``requires_grad=False``), and routes
inference through :func:`torch.no_grad`. The frozen-parameter assertion
is the runtime substrate the ``_FORBIDDEN_ATTRS`` shield (plan/04 §3.3)
backstops at the attribute-name level.

The Phase-0 MAPPO actor is the tiny shared-parameter MLP plan/04 §3.5
calls for: two Tanh hidden layers + a linear action head. Layer widths
are read off the state-dict tensors so the wrapper does not need an
architecture-metadata sidecar (the existing
:class:`~concerto.training.checkpoints.CheckpointMetadata` carries
provenance only — adding architecture fields would couple every
checkpoint format to this one).

Lazy load: heavy I/O is deferred until the first :meth:`reset` or
:meth:`act` call so :class:`chamber.partners.registry.load_partner`
remains a cheap registry lookup. ``artifacts_root`` defaults to
``./artifacts`` (matching :attr:`concerto.training.config.EgoAHTConfig.artifacts_root`)
and can be overridden via the ``CONCERTO_ARTIFACTS_ROOT`` environment
variable so tests + the B4 zoo-seed reproduction can point at a
fixture directory without monkey-patching the partner.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from chamber.partners.interface import PartnerBase
from chamber.partners.registry import register_partner
from concerto.training.checkpoints import load_checkpoint

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray

    from chamber.partners.api import PartnerSpec

#: Environment variable consulted to override the default ``artifacts_root``.
#:
#: Matches the convention used by the Phase-0 zoo-seed reproduction path
#: (plan/05 §6 #5). Tests set this via ``monkeypatch.setenv`` so the
#: ``local://artifacts/...`` URI resolves under ``tmp_path``.
_ARTIFACTS_ROOT_ENV: str = "CONCERTO_ARTIFACTS_ROOT"

#: Default ``artifacts_root`` when the env var is unset.
#:
#: Matches :attr:`concerto.training.config.EgoAHTConfig.artifacts_root`'s
#: default of ``Path("./artifacts")``; both resolve relative to the
#: current working directory at load time, not at import time.
_DEFAULT_ARTIFACTS_ROOT: str = "./artifacts"

#: Sub-key inside the loaded checkpoint dict that holds the actor's
#: tensors (ADR-002 §Decisions; mirrors
#: :meth:`chamber.benchmarks.ego_ppo_trainer.EgoPPOTrainer.state_dict`).
#:
#: The Phase-0 MAPPO checkpoint format is ``{"actor": <flat_state_dict>}``
#: — the same outer shape as the HARL-side trainer's ``state_dict`` so
#: the two frozen-RL adapters share a wire format. Direct (un-nested)
#: state-dicts are rejected (ADR-002 §Decisions: one canonical wire
#: format per checkpoint type).
_ACTOR_KEY: str = "actor"

#: Weight-tensor keys the wrapper inspects to infer layer widths.
#:
#: A MAPPO actor saved by this module produces ``fc1.weight``,
#: ``fc1.bias``, ``fc2.weight``, ``fc2.bias``, ``head.weight``,
#: ``head.bias`` per :class:`_MAPPOActor`. The bias tensors are
#: required by :meth:`torch.nn.Module.load_state_dict`; the constants
#: below only name the weight keys because shape inference uses
#: ``fc1.weight.shape`` and ``head.weight.shape``. Renaming a layer in
#: :class:`_MAPPOActor` is a serialisation break — update both sides.
_FC1_WEIGHT_KEY: str = "fc1.weight"
_FC2_WEIGHT_KEY: str = "fc2.weight"
_HEAD_WEIGHT_KEY: str = "head.weight"


class _MAPPOActor(nn.Module):
    """Tiny shared-parameter MLP actor (plan/04 §3.5; ADR-009 §Decision).

    Two Tanh hidden layers + a linear action head. Tanh matches the
    activation used in HAPPO's hand-rolled critic (plan/05 §3.5) so the
    two frozen-partner adapters share an activation convention.

    The class is private to :mod:`chamber.partners.frozen_mappo`; external
    callers go through :class:`FrozenMAPPOPartner` which handles loading,
    freezing, and the no-grad inference contract.
    """

    def __init__(self, *, obs_dim: int, hidden_dim: int, action_dim: int) -> None:
        """Build the actor MLP (plan/04 §3.5).

        Args:
            obs_dim: Length of the flat observation vector
                (``obs["agent"][uid]["state"]``).
            hidden_dim: Width of each hidden layer.
            action_dim: Length of the output action vector for this uid.
        """
        super().__init__()
        self.fc1: nn.Linear = nn.Linear(obs_dim, hidden_dim)
        self.fc2: nn.Linear = nn.Linear(hidden_dim, hidden_dim)
        self.head: nn.Linear = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, obs_dim)`` → ``(batch, action_dim)`` (ADR-009 §Decision; plan/04 §3.5)."""
        hidden = torch.tanh(self.fc1(obs))
        hidden = torch.tanh(self.fc2(hidden))
        return self.head(hidden)


@register_partner("frozen_mappo")
class FrozenMAPPOPartner(PartnerBase):
    """Frozen shared-parameter MAPPO partner (ADR-009 §Decision; plan/04 §3.5).

    Loads a :class:`_MAPPOActor` checkpoint via
    :func:`concerto.training.checkpoints.load_checkpoint` (which verifies
    the sidecar SHA-256), freezes every parameter
    (``requires_grad=False``), and runs inference under
    :func:`torch.no_grad`. The actor's layer widths are inferred from the
    loaded state-dict tensor shapes — no architecture-metadata sidecar is
    required (ADR-002 §Decisions: provenance lives on the sidecar; shape
    lives in the tensors).

    Spec contract (plan/04 §3.5; ADR-009 §Decision):

    - ``spec.class_name == "frozen_mappo"``.
    - ``spec.weights_uri`` is the ``local://artifacts/<name>.pt`` URI.
    - ``spec.extra["uid"]`` is the env-side uid the partner acts on;
      it indexes ``obs["agent"][uid]["state"]`` at action time.

    Statefulness (plan/04 §2): the partner is stateless across episodes;
    :meth:`reset` is a no-op once the checkpoint is loaded. Loading is
    deferred to the first :meth:`reset` / :meth:`act` call so the
    registry lookup in :func:`chamber.partners.registry.load_partner`
    stays cheap and so test fixtures can construct + monkey-patch the
    env var before any I/O happens.
    """

    def __init__(self, spec: PartnerSpec) -> None:
        """Bind the spec; defer checkpoint I/O (ADR-009 §Decision; plan/04 §3.3).

        Args:
            spec: Identity-bearing handle. Reads ``spec.extra["uid"]``
                for the env-side agent uid and ``spec.weights_uri`` for
                the checkpoint URI. Construction does not touch disk —
                the checkpoint is loaded lazily on first action / reset.

        Raises:
            ValueError: If ``spec.weights_uri`` is ``None`` or
                ``spec.extra["uid"]`` is missing / empty.
        """
        super().__init__(spec)
        if spec.weights_uri is None:
            msg = (
                f"FrozenMAPPOPartner requires spec.weights_uri (ADR-009 §Decision); "
                f"got None on spec {spec.class_name!r}."
            )
            raise ValueError(msg)
        uid = spec.extra.get("uid", "")
        if not uid:
            msg = (
                "FrozenMAPPOPartner requires spec.extra['uid'] (the env-side agent uid; "
                "plan/04 §3.5)."
            )
            raise ValueError(msg)
        self._uid: str = uid
        self._weights_uri: str = spec.weights_uri
        self._actor: _MAPPOActor | None = None

    def reset(self, *, seed: int | None = None) -> None:
        """Ensure the checkpoint is loaded; no per-episode state (ADR-009 §Decision; plan/04 §2).

        Args:
            seed: Accepted for :class:`~chamber.partners.api.FrozenPartner`
                Protocol conformance. The partner is fully deterministic
                given the loaded weights and ``obs``; seed is ignored.
        """
        del seed
        self._ensure_loaded()

    def act(
        self,
        obs: Mapping[str, object],
        *,
        deterministic: bool = True,
    ) -> NDArray[np.floating]:
        """Greedy MAPPO-actor forward pass under no-grad (ADR-009 §Decision).

        Args:
            obs: Gymnasium observation mapping. Reads the ego's flat state
                from ``obs["agent"][uid]["state"]`` (plan/04 §3.4); other
                keys are ignored.
            deterministic: Accepted for Protocol conformance. The
                Phase-0 actor head is a deterministic linear layer with
                no sampled noise, so this kwarg has no effect.

        Returns:
            ``np.float32`` action vector of length ``action_dim`` (the
            head's output dimension, inferred from the loaded
            checkpoint).

        Raises:
            ValueError: If ``obs["agent"][uid]["state"]`` is missing or
                its length disagrees with the loaded actor's input
                dimension.
        """
        del deterministic
        actor = self._ensure_loaded()
        state = _read_flat_state(obs, self._uid)
        expected = actor.fc1.in_features
        if state.shape != (expected,):
            msg = (
                f"FrozenMAPPOPartner: obs['agent'][{self._uid!r}]['state'] has shape "
                f"{tuple(state.shape)}, expected ({expected},) matching the loaded "
                f"actor's fc1 input dim."
            )
            raise ValueError(msg)
        with torch.no_grad():
            input_t = torch.from_numpy(state).unsqueeze(0)
            action_t = actor(input_t)
        return action_t.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def _ensure_loaded(self) -> _MAPPOActor:
        """Load + freeze the actor on first use; idempotent (plan/04 §3.3).

        Returns:
            The loaded :class:`_MAPPOActor` (cached after first call).

        Raises:
            CheckpointError: Propagated from
                :func:`concerto.training.checkpoints.load_checkpoint`
                when the sidecar is missing, malformed, or the SHA-256
                disagrees with the on-disk payload.
            ValueError: When the loaded state-dict does not match the
                expected MAPPO actor layout.
        """
        if self._actor is not None:
            return self._actor
        artifacts_root = _resolve_artifacts_root()
        loaded, _meta = load_checkpoint(uri=self._weights_uri, artifacts_root=artifacts_root)
        actor = self._build_actor_from_state_dict(loaded)
        for param in actor.parameters():
            param.requires_grad = False
        actor.eval()
        self._actor = actor
        return actor

    def _build_actor_from_state_dict(self, loaded: Mapping[str, object]) -> _MAPPOActor:
        """Reconstruct the MLP actor from a loaded checkpoint dict (plan/04 §3.5).

        Accepts both nested (``{"actor": <flat_state_dict>}``) and flat
        layouts. The nested layout is the canonical Phase-0 format
        produced by :meth:`chamber.benchmarks.ego_ppo_trainer.EgoPPOTrainer.state_dict`
        (the ``"actor"`` sub-key); the flat layout is the convenience
        format the unit tests + manual reproduction scripts use.

        Args:
            loaded: The dict :func:`load_checkpoint` returned.

        Returns:
            A fresh :class:`_MAPPOActor` with weights copied from
            ``loaded``.

        Raises:
            ValueError: When required layer keys are missing or shapes
                disagree.
        """
        state_dict = _extract_actor_state_dict(loaded)
        try:
            fc1_weight = state_dict[_FC1_WEIGHT_KEY]
            fc2_weight = state_dict[_FC2_WEIGHT_KEY]
            head_weight = state_dict[_HEAD_WEIGHT_KEY]
        except KeyError as exc:
            msg = (
                f"FrozenMAPPOPartner: loaded actor state-dict is missing required "
                f"layer key {exc.args[0]!r}; expected the fc1/fc2/head MLP layout "
                f"(plan/04 §3.5). Got keys: {sorted(state_dict.keys())}."
            )
            raise ValueError(msg) from exc
        if not (
            isinstance(fc1_weight, torch.Tensor)
            and isinstance(fc2_weight, torch.Tensor)
            and isinstance(head_weight, torch.Tensor)
        ):
            msg = (
                "FrozenMAPPOPartner: actor state-dict layer values must be "
                "torch.Tensor; got non-tensor entries."
            )
            raise ValueError(msg)
        hidden_dim, obs_dim = int(fc1_weight.shape[0]), int(fc1_weight.shape[1])
        action_dim = int(head_weight.shape[0])
        if fc2_weight.shape != (hidden_dim, hidden_dim):
            msg = (
                f"FrozenMAPPOPartner: fc2 shape {tuple(fc2_weight.shape)} disagrees "
                f"with inferred hidden_dim {hidden_dim} (plan/04 §3.5)."
            )
            raise ValueError(msg)
        if head_weight.shape[1] != hidden_dim:
            msg = (
                f"FrozenMAPPOPartner: head input dim {head_weight.shape[1]} disagrees "
                f"with inferred hidden_dim {hidden_dim} (plan/04 §3.5)."
            )
            raise ValueError(msg)
        actor = _MAPPOActor(obs_dim=obs_dim, hidden_dim=hidden_dim, action_dim=action_dim)
        try:
            actor.load_state_dict(dict(state_dict))
        except RuntimeError as exc:
            msg = f"FrozenMAPPOPartner: load_state_dict failed for {self._weights_uri!r}: {exc}"
            raise ValueError(msg) from exc
        return actor


def _extract_actor_state_dict(
    loaded: Mapping[str, object],
) -> Mapping[str, object]:
    """Pull the actor sub-dict out of a loaded checkpoint (plan/04 §3.5).

    The canonical Phase-0 layout is ``{"actor": <flat_state_dict>}`` —
    matching the HARL-side B2 checkpoint shape and the
    :meth:`EgoPPOTrainer.state_dict` keying. Non-tensor entries are
    NOT filtered out — :meth:`FrozenMAPPOPartner._build_actor_from_state_dict`
    flags them with the correct error message ("layer values must be
    torch.Tensor") instead of pretending the key was absent.

    Args:
        loaded: The dict :func:`load_checkpoint` returned.

    Returns:
        The actor's state-dict (the value of ``loaded["actor"]``).

    Raises:
        ValueError: When ``loaded`` does not carry an ``"actor"`` key
            that maps to a dict.
    """
    inner = loaded.get(_ACTOR_KEY)
    if isinstance(inner, dict):
        return inner
    msg = (
        f"FrozenMAPPOPartner: loaded checkpoint does not contain a dict under "
        f"the {_ACTOR_KEY!r} key. Got top-level keys: {sorted(loaded.keys())}."
    )
    raise ValueError(msg)


def _read_flat_state(obs: Mapping[str, object], uid: str) -> NDArray[np.float32]:
    """Extract the flat ego state vector from a nested obs dict (plan/04 §3.4).

    Mirrors :func:`chamber.benchmarks.ego_ppo_trainer._flat_ego_obs` but
    raises a partner-flavoured :class:`ValueError` on layout mismatch
    rather than letting a ``KeyError`` from ``obs["agent"][uid]`` leak
    out of the partner.

    Args:
        obs: Gymnasium observation mapping.
        uid: Partner's own env-side uid.

    Returns:
        ``np.float32`` 1-D array of the partner's flat state.

    Raises:
        ValueError: If ``obs["agent"][uid]["state"]`` is absent or not
            a 1-D array.
    """
    agent = obs.get("agent")
    if not isinstance(agent, dict) or uid not in agent:
        msg = (
            f"FrozenMAPPOPartner: obs['agent'][{uid!r}] missing — the partner cannot "
            f"act without its proprioceptive state (plan/04 §3.4)."
        )
        raise ValueError(msg)
    entry = agent[uid]
    if not isinstance(entry, dict) or "state" not in entry:
        msg = f"FrozenMAPPOPartner: obs['agent'][{uid!r}]['state'] missing (plan/04 §3.4)."
        raise ValueError(msg)
    state = np.asarray(entry["state"], dtype=np.float32)
    if state.ndim != 1:
        msg = (
            f"FrozenMAPPOPartner: obs['agent'][{uid!r}]['state'] must be 1-D; "
            f"got shape {state.shape}."
        )
        raise ValueError(msg)
    return state


def _resolve_artifacts_root() -> Path:
    """Resolve the artefact root for ``local://artifacts/...`` URIs (plan/05 §6 #5).

    Reads the ``CONCERTO_ARTIFACTS_ROOT`` env var when set; otherwise
    falls back to ``./artifacts`` (matching
    :attr:`concerto.training.config.EgoAHTConfig.artifacts_root`).
    Resolution is per-call rather than module-load so tests can flip
    the env var via ``monkeypatch.setenv`` between cases.

    Returns:
        The artefact root as a :class:`~pathlib.Path`.
    """
    raw = os.environ.get(_ARTIFACTS_ROOT_ENV, _DEFAULT_ARTIFACTS_ROOT)
    return Path(raw)


__all__ = ["FrozenMAPPOPartner"]
