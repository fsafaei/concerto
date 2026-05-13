# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false
#
# Same rationale as :mod:`chamber.benchmarks.ego_ppo_trainer`: torch's
# stub files do not export ``no_grad`` / ``from_numpy`` via ``__all__``
# even though they are public API per the official docs. Suppress
# file-locally so per-line ``type: ignore`` clutter stays out.
"""Frozen HARL HAPPO partner — Stage-3 draft-zoo entry #3 (ADR-009 §Decision; plan/04 §3.6).

The wrapper loads the HARL HAPPO actor checkpoint produced by
:class:`chamber.benchmarks.ego_ppo_trainer.EgoPPOTrainer` (which saves
``{"actor": <state_dict>, "actor_optim": ..., "critic": ..., "critic_optim": ...}``
via :func:`concerto.training.checkpoints.save_checkpoint`), rebuilds the
HARL :class:`StochasticPolicy` with the same hyperparameters the trainer
used, loads the ``"actor"`` sub-dict, freezes every parameter, and runs
inference under :func:`torch.no_grad` with ``deterministic=True``.

Shape inference (plan/04 §3.6; matches the
:class:`chamber.partners.frozen_mappo.FrozenMAPPOPartner` convention):
``obs_dim`` is read from ``base.feature_norm.weight``, ``hidden_dim`` from
``base.mlp.fc.0.weight``, and ``action_dim`` from
``act.action_out.fc_mean.weight``. The wrapper does not need an
architecture-metadata sidecar — provenance lives on
:class:`~concerto.training.checkpoints.CheckpointMetadata`, shape lives
in the tensors.

The HARL-specific arguments (activation, init, std coefficients,
feature-normalisation flag) are pinned to the project-wide defaults from
:data:`chamber.benchmarks.ego_ppo_trainer._HARL_FIXED_DEFAULTS`; any
divergence between training-time and load-time args surfaces as a
:class:`ValueError` at :meth:`StochasticPolicy.load_state_dict` time.
Two-layer MLP is assumed (matches every Phase-0 trainer config); deeper
MLPs would need ``hidden_sizes`` carried in spec metadata and a
follow-up ADR.

Determinism (P6 / ADR-002 §Decisions): HARL's :class:`DiagGaussian`
samples actions from torch's global RNG. Honouring ``deterministic=False``
in :meth:`act` would couple a frozen partner's output to whatever else
in the process touched the global generator. Phase-0 pins the partner
to ``deterministic=True`` regardless of the caller-supplied kwarg —
match :class:`~chamber.partners.frozen_mappo.FrozenMAPPOPartner`'s
behaviour. The kwarg is accepted for
:class:`~chamber.partners.api.FrozenPartner` Protocol conformance and
its no-op status is pinned by :class:`TestActionContract`.

Lazy load: heavy I/O is deferred until the first :meth:`reset` or
:meth:`act` call so :class:`chamber.partners.registry.load_partner`
stays a cheap registry lookup. ``artifacts_root`` defaults to
``./artifacts`` (matching
:attr:`concerto.training.config.EgoAHTConfig.artifacts_root`) and can be
overridden via the ``CONCERTO_ARTIFACTS_ROOT`` environment variable
(see ADR-002 §Revision-history footnote 2026-05-13).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np
import torch
from harl.models.policy_models.stochastic_policy import StochasticPolicy

from chamber.partners.interface import PartnerBase
from chamber.partners.registry import register_partner
from concerto.training.checkpoints import load_checkpoint

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray

    from chamber.partners.api import PartnerSpec

#: Environment variable consulted to override the default ``artifacts_root``.
#:
#: Matches :mod:`chamber.partners.frozen_mappo`'s convention (ADR-002
#: §Revision-history footnote 2026-05-13).
_ARTIFACTS_ROOT_ENV: str = "CONCERTO_ARTIFACTS_ROOT"

#: Default ``artifacts_root`` when the env var is unset (matches
#: :attr:`concerto.training.config.EgoAHTConfig.artifacts_root`).
_DEFAULT_ARTIFACTS_ROOT: str = "./artifacts"

#: Sub-key inside the loaded checkpoint dict that holds the actor's
#: state-dict (matches :meth:`EgoPPOTrainer.state_dict`'s output shape).
_ACTOR_KEY: str = "actor"

#: HARL :class:`StochasticPolicy` state-dict keys the wrapper inspects.
#:
#: Layer naming mirrors HARL's actor topology:
#:
#: - ``base.feature_norm.*`` — :class:`torch.nn.LayerNorm` on the input
#:   (present iff ``args["use_feature_normalization"]`` is ``True``;
#:   the project's default is ``True``).
#: - ``base.mlp.fc.0.*`` — first :class:`torch.nn.Linear` (input → hidden).
#: - ``base.mlp.fc.3.*`` — second :class:`torch.nn.Linear` (hidden → hidden;
#:   layers 1 and 2 inside the ``nn.Sequential`` are activation + LayerNorm
#:   and carry no parameters relevant to shape inference here).
#: - ``act.action_out.fc_mean.*`` — :class:`torch.nn.Linear` for the
#:   :class:`DiagGaussian` mean head (hidden → action).
#:
#: Renaming any of these in the HARL fork is a serialisation break — update
#: both sides in the same ADR amendment.
_FEATURE_NORM_WEIGHT_KEY: str = "base.feature_norm.weight"
_FC0_WEIGHT_KEY: str = "base.mlp.fc.0.weight"
_FC_MEAN_WEIGHT_KEY: str = "act.action_out.fc_mean.weight"

#: HARL :class:`StochasticPolicy` constructor args used at load time.
#:
#: Mirrors the subset of
#: :data:`chamber.benchmarks.ego_ppo_trainer._HARL_FIXED_DEFAULTS` that
#: :class:`StochasticPolicy` / :class:`MLPBase` / :class:`ACTLayer` /
#: :class:`DiagGaussian` actually read at construction. Project-wide
#: defaults — drifting between training-time and load-time args produces
#: a state-dict mismatch at :meth:`load_state_dict`, which the wrapper
#: catches and re-raises as :class:`ValueError`.
#:
#: ``hidden_sizes`` is overwritten per-load with the value inferred from
#: the checkpoint's ``base.mlp.fc.0.weight`` shape.
_HARL_INFERENCE_ARGS: dict[str, Any] = {
    "hidden_sizes": [128, 128],  # placeholder; overwritten per-checkpoint.
    "use_feature_normalization": True,
    "activation_func": "relu",
    "initialization_method": "orthogonal_",
    "use_recurrent_policy": False,
    "use_naive_recurrent_policy": False,
    "use_policy_active_masks": False,
    "recurrent_n": 1,
    "gain": 0.01,
    "std_x_coef": 1.0,
    "std_y_coef": 0.5,
}


@register_partner("frozen_harl")
class FrozenHARLPartner(PartnerBase):
    """Frozen HARL HAPPO partner (ADR-009 §Decision; plan/04 §3.6).

    Loads a HARL :class:`StochasticPolicy` checkpoint via
    :func:`concerto.training.checkpoints.load_checkpoint` (which verifies
    the sidecar SHA-256), freezes every parameter
    (``requires_grad=False``), and runs inference under
    :func:`torch.no_grad` with ``deterministic=True``. The actor's layer
    widths are inferred from the loaded state-dict tensor shapes.

    Spec contract (plan/04 §3.6; ADR-009 §Decision):

    - ``spec.class_name == "frozen_harl"``.
    - ``spec.weights_uri`` is the ``local://artifacts/<name>.pt`` URI.
    - ``spec.extra["uid"]`` is the env-side uid the partner acts on;
      it indexes ``obs["agent"][uid]["state"]`` at action time.

    Statefulness (plan/04 §2): the partner is stateless across episodes;
    :meth:`reset` is a no-op once the checkpoint is loaded. Loading is
    deferred to the first :meth:`reset` / :meth:`act` call so the
    registry lookup in :func:`chamber.partners.registry.load_partner`
    stays cheap.
    """

    def __init__(self, spec: PartnerSpec) -> None:
        """Bind the spec; defer checkpoint I/O (ADR-009 §Decision; plan/04 §3.3).

        Args:
            spec: Identity-bearing handle. Reads ``spec.extra["uid"]``
                for the env-side agent uid and ``spec.weights_uri`` for
                the checkpoint URI. Construction does not touch disk.

        Raises:
            ValueError: If ``spec.weights_uri`` is ``None`` or
                ``spec.extra["uid"]`` is missing / empty.
        """
        super().__init__(spec)
        if spec.weights_uri is None:
            msg = (
                f"FrozenHARLPartner requires spec.weights_uri (ADR-009 §Decision); "
                f"got None on spec {spec.class_name!r}."
            )
            raise ValueError(msg)
        uid = spec.extra.get("uid", "")
        if not uid:
            msg = (
                "FrozenHARLPartner requires spec.extra['uid'] (the env-side agent uid; "
                "plan/04 §3.6)."
            )
            raise ValueError(msg)
        self._uid: str = uid
        self._weights_uri: str = spec.weights_uri
        self._actor: StochasticPolicy | None = None
        self._obs_dim: int = 0
        self._action_dim: int = 0
        self._hidden_dim: int = 0

    def reset(self, *, seed: int | None = None) -> None:
        """Ensure the checkpoint is loaded; no per-episode state (ADR-009 §Decision; plan/04 §2).

        Args:
            seed: Accepted for
                :class:`~chamber.partners.api.FrozenPartner` Protocol
                conformance. The partner is fully deterministic given
                the loaded weights and ``obs``; seed is ignored.
        """
        del seed
        self._ensure_loaded()

    def act(
        self,
        obs: Mapping[str, object],
        *,
        deterministic: bool = True,
    ) -> NDArray[np.floating]:
        """Greedy HARL-actor forward pass under no-grad (ADR-009 §Decision).

        Args:
            obs: Gymnasium observation mapping. Reads the ego's flat state
                from ``obs["agent"][uid]["state"]`` (plan/04 §3.4); other
                keys are ignored.
            deterministic: Accepted for Protocol conformance but ignored.
                Phase-0 pins the partner to the distribution mode so a
                frozen partner's actions stay decoupled from torch's
                global RNG (see module docstring).

        Returns:
            ``np.float32`` action vector of length ``action_dim`` (the
            mean head's output dimension, inferred from the loaded
            checkpoint).

        Raises:
            ValueError: If ``obs["agent"][uid]["state"]`` is missing or
                its length disagrees with the loaded actor's input
                dimension.
        """
        del deterministic
        actor = self._ensure_loaded()
        state = _read_flat_state(obs, self._uid)
        if state.shape != (self._obs_dim,):
            msg = (
                f"FrozenHARLPartner: obs['agent'][{self._uid!r}]['state'] has shape "
                f"{tuple(state.shape)}, expected ({self._obs_dim},) matching the loaded "
                f"actor's input dim."
            )
            raise ValueError(msg)
        # HARL's StochasticPolicy.forward signature is
        # ``(obs, rnn_states, masks, available_actions, deterministic)``.
        # rnn_states + masks are placeholders since use_recurrent_policy=False;
        # we pass shape-correct zeros / ones so HARL's internal ``check`` call
        # accepts them. Mirrors EgoPPOTrainer.act's call site.
        obs_t = state.reshape(1, -1)
        rnn_states = np.zeros((1, 1, self._hidden_dim), dtype=np.float32)
        masks = np.ones((1, 1), dtype=np.float32)
        with torch.no_grad():
            action_t, _, _ = actor(obs_t, rnn_states, masks, None, True)
        return action_t.detach().cpu().numpy().squeeze(0).astype(np.float32)

    def _ensure_loaded(self) -> StochasticPolicy:
        """Load + freeze the HARL actor on first use; idempotent (plan/04 §3.3).

        Returns:
            The loaded :class:`StochasticPolicy` (cached after first call).

        Raises:
            CheckpointError: Propagated from
                :func:`concerto.training.checkpoints.load_checkpoint`
                when the sidecar is missing, malformed, or the SHA-256
                disagrees with the on-disk payload.
            ValueError: When the loaded state-dict does not match the
                expected HARL HAPPO actor layout.
        """
        if self._actor is not None:
            return self._actor
        artifacts_root = _resolve_artifacts_root()
        loaded, _meta = load_checkpoint(uri=self._weights_uri, artifacts_root=artifacts_root)
        actor_sd = _extract_actor_state_dict(loaded)
        obs_dim, hidden_dim, action_dim = _infer_shape(actor_sd)
        args = dict(_HARL_INFERENCE_ARGS)
        args["hidden_sizes"] = [hidden_dim, hidden_dim]
        obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
        actor = StochasticPolicy(args, obs_space, act_space, torch.device("cpu"))
        try:
            actor.load_state_dict(dict(actor_sd))
        except RuntimeError as exc:
            msg = f"FrozenHARLPartner: load_state_dict failed for {self._weights_uri!r}: {exc}"
            raise ValueError(msg) from exc
        for param in actor.parameters():
            param.requires_grad = False
        actor.eval()
        self._actor = actor
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._hidden_dim = hidden_dim
        return actor


def _extract_actor_state_dict(
    loaded: Mapping[str, object],
) -> Mapping[str, object]:
    """Pull the actor sub-dict out of a loaded HARL checkpoint (plan/04 §3.6).

    The canonical Phase-0 layout is the full
    :meth:`EgoPPOTrainer.state_dict` output:
    ``{"actor": ..., "actor_optim": ..., "critic": ..., "critic_optim": ...}``.
    The wrapper only reads the ``"actor"`` sub-key; the
    optimizer + critic entries are ignored.

    Args:
        loaded: The dict :func:`load_checkpoint` returned.

    Returns:
        The actor's state-dict.

    Raises:
        ValueError: When ``loaded`` does not carry an ``"actor"`` key
            that maps to a dict.
    """
    inner = loaded.get(_ACTOR_KEY)
    if isinstance(inner, dict):
        return inner
    msg = (
        f"FrozenHARLPartner: loaded checkpoint does not contain a dict under the "
        f"{_ACTOR_KEY!r} key. Got top-level keys: {sorted(loaded.keys())}."
    )
    raise ValueError(msg)


def _infer_shape(actor_sd: Mapping[str, object]) -> tuple[int, int, int]:
    """Infer ``(obs_dim, hidden_dim, action_dim)`` from the actor state-dict (plan/04 §3.6).

    Reads:

    - ``base.feature_norm.weight.shape[0]`` → ``obs_dim``.
    - ``base.mlp.fc.0.weight.shape[0]`` → ``hidden_dim``.
    - ``act.action_out.fc_mean.weight.shape[0]`` → ``action_dim``.

    The wrapper assumes a 2-hidden-layer MLP topology with feature
    normalisation enabled (the project's
    :data:`~chamber.benchmarks.ego_ppo_trainer._HARL_FIXED_DEFAULTS`
    contract). Checkpoints saved without feature normalisation are
    rejected at this step.

    Args:
        actor_sd: The actor sub-dict returned by :func:`_extract_actor_state_dict`.

    Returns:
        ``(obs_dim, hidden_dim, action_dim)`` tuple of positive ints.

    Raises:
        ValueError: When any of the three required tensors is absent or
            non-tensor.
    """
    obs_dim = _read_tensor_shape(actor_sd, _FEATURE_NORM_WEIGHT_KEY, dim=0)
    hidden_dim = _read_tensor_shape(actor_sd, _FC0_WEIGHT_KEY, dim=0)
    action_dim = _read_tensor_shape(actor_sd, _FC_MEAN_WEIGHT_KEY, dim=0)
    return obs_dim, hidden_dim, action_dim


def _read_tensor_shape(sd: Mapping[str, object], key: str, *, dim: int) -> int:
    """Read ``sd[key].shape[dim]`` with partner-flavoured error messages (plan/04 §3.6)."""
    value = sd.get(key)
    if value is None:
        msg = (
            f"FrozenHARLPartner: actor state-dict is missing required key {key!r}; "
            f"shape inference requires the HARL StochasticPolicy layout from "
            f"chamber.benchmarks.ego_ppo_trainer (plan/04 §3.6)."
        )
        raise ValueError(msg)
    if not isinstance(value, torch.Tensor):
        msg = (
            f"FrozenHARLPartner: actor state-dict value at {key!r} must be a "
            f"torch.Tensor; got {type(value).__name__}."
        )
        raise ValueError(msg)
    return int(value.shape[dim])


def _read_flat_state(obs: Mapping[str, object], uid: str) -> NDArray[np.float32]:
    """Extract the flat ego state vector from a nested obs dict (plan/04 §3.4)."""
    agent = obs.get("agent")
    if not isinstance(agent, dict) or uid not in agent:
        msg = (
            f"FrozenHARLPartner: obs['agent'][{uid!r}] missing — the partner cannot "
            f"act without its proprioceptive state (plan/04 §3.4)."
        )
        raise ValueError(msg)
    entry = agent[uid]
    if not isinstance(entry, dict) or "state" not in entry:
        msg = f"FrozenHARLPartner: obs['agent'][{uid!r}]['state'] missing (plan/04 §3.4)."
        raise ValueError(msg)
    state = np.asarray(entry["state"], dtype=np.float32)
    if state.ndim != 1:
        msg = (
            f"FrozenHARLPartner: obs['agent'][{uid!r}]['state'] must be 1-D; "
            f"got shape {state.shape}."
        )
        raise ValueError(msg)
    return state


def _resolve_artifacts_root() -> Path:
    """Resolve the artefact root for ``local://artifacts/...`` URIs (plan/05 §6 #5).

    Reads the ``CONCERTO_ARTIFACTS_ROOT`` env var when set; otherwise
    falls back to ``./artifacts``. Per-call so tests can flip the env
    var via ``monkeypatch.setenv`` between cases (matches the
    :mod:`chamber.partners.frozen_mappo` helper).
    """
    raw = os.environ.get(_ARTIFACTS_ROOT_ENV, _DEFAULT_ARTIFACTS_ROOT)
    return Path(raw)


__all__ = ["FrozenHARLPartner"]
