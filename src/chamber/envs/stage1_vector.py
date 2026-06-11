# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false
#
# Same suppression rationale as :mod:`chamber.envs.stage1_pickplace`:
# ``torch.as_tensor`` / ``torch.int32`` are canonical public API but not
# advertised in the stub's ``__all__``. Suppressed file-locally.
"""Vector auto-reset wrapper for the Stage-1b GPU-parallel cell (ADR-007 §Stage 1b).

P1.05.10 regime-alignment slice. ManiSkill v3's :class:`BaseEnv` at
``num_envs > 1`` supports per-env partial reset
(``reset(options={"env_idx": ...})``) but does not auto-reset done envs
inside ``step`` — that is normally supplied by ManiSkill's own gym
vector wrapper, which a *directly*-instantiated ``BaseEnv`` subclass
(the ADR-001 wrapper-only pattern) does not have. This module ships the
chamber-side analogue so the vectorised training loop
(:func:`concerto.training.ego_aht.train` batched path) sees standard
vector semantics:

- ``step`` returns batched ``(obs, reward, terminated, truncated, info)``
  with done envs **already reset** to a fresh episode;
- the pre-reset final observation of the step is surfaced as
  ``info["final_observation"]`` (full batched obs dict) so the trainer
  can compute the Pardo-2017 truncation bootstrap
  ``V(s_truncated_final)`` against the *actual* terminal obs rather
  than the new episode's reset obs (issue #62 / Pardo 2017 §4 — the
  same contract the single-env loop honours via its eager critic call
  at the boundary).

Episode randomisation determinism (P6 / ADR-002): the wrapper passes no
seed to the partial reset — per-episode cube/goal draws come from the
inner env's per-env-index ``derive_substream`` streams (see
:class:`chamber.envs.stage1_pickplace.Stage1PickPlaceEnv`), advanced
only when that env resets, so reset interleaving across the batch
cannot perturb any other env's stream.

Tier-1 contract (ADR-001): no SAPIEN / ManiSkill imports at module top
level; torch is imported lazily inside the step body (the wrapper also
accepts numpy-flagged Tier-1 fake envs so the batched-loop unit tests
run on a Vulkan-less host).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

__all__ = ["Stage1AutoResetWrapper"]


def _to_bool_array(flags: Any) -> np.ndarray:  # type: ignore[type-arg]  # noqa: ANN401 - torch.Tensor or np.ndarray depending on env backend
    """Coerce a batched done-flag tensor/array to a 1-D numpy bool array (ADR-007 §Stage 1b)."""
    if hasattr(flags, "detach"):
        flags = flags.detach().cpu().numpy()
    return np.asarray(flags).reshape(-1).astype(bool)


class Stage1AutoResetWrapper(gym.Wrapper):  # type: ignore[type-arg]
    """Per-env auto-reset for vectorised Stage-1b cells (ADR-007 §Stage 1b; P1.05.10).

    Applied outermost by
    :func:`chamber.envs.stage1_pickplace.make_stage1_pickplace_env` when
    ``num_envs > 1``. On every ``step`` where any env reports
    ``terminated | truncated``, the wrapper:

    1. copies ``info`` and attaches the pre-reset batched obs as
       ``info["final_observation"]`` (consumed by
       :meth:`chamber.benchmarks.ego_ppo_trainer.EgoPPOTrainer.observe`
       for the Pardo truncation bootstrap);
    2. partially resets exactly the done envs via
       ``self.env.reset(options={"env_idx": <done indices>})`` and
       returns the post-reset obs in place of the pre-reset one.

    The ``terminated`` / ``truncated`` flags returned to the caller are
    the *pre-reset* flags — the training loop reads them for episode
    accounting and GAE boundary handling, exactly as a Gymnasium
    vector env's autoreset contract does.

    Single-env cells never see this wrapper (ADR-002: the historical
    ``num_envs == 1`` loop, which owns its own ``reset(seed=...)``
    cadence, is byte-identical to pre-P1.05.10 builds).
    """

    def step(  # type: ignore[override]
        self,
        action: Any,  # noqa: ANN401 - ManiSkill dict-of-batched-tensors action
    ) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        """Step all envs; partial-reset the done subset (ADR-007 §Stage 1b; P1.05.10)."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = _to_bool_array(terminated) | _to_bool_array(truncated)
        if bool(done.any()):
            info = dict(info)
            info["final_observation"] = obs
            done_idx = np.nonzero(done)[0]
            reset_obs, _ = self.env.reset(options={"env_idx": _as_env_idx(done_idx, obs)})
            obs = reset_obs
        return obs, reward, terminated, truncated, info

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 - mirrors the Stage1 obs wrappers' forwarding contract
        """Forward attribute access to the inner env (Tier-2 caller contract; ADR-007 §Stage 1b).

        Mirrors :meth:`chamber.envs.stage1_obs_filter.Stage1ASStateSynthesizer.__getattr__`:
        Gymnasium 1.3 removed :class:`gym.Wrapper`'s implicit forwarding,
        and callers of the factory read inner-env attributes
        (``env.agent``, ``env.condition_config``, ``env.ego_uid``…)
        through the wrapper chain.
        """
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)


def _as_env_idx(done_idx: np.ndarray, obs: Any) -> Any:  # type: ignore[type-arg]  # noqa: ANN401 - returns torch.Tensor on torch-backed envs, ndarray otherwise
    """Coerce done indices to the inner env's expected ``env_idx`` type (ADR-007 §Stage 1b).

    ManiSkill GPU envs expect a torch tensor (matched to the obs
    device); Tier-1 fake envs take the numpy array as-is. The obs is
    used only to sniff the backend (any torch tensor in the batch ⇒
    torch path).
    """
    try:
        import torch  # noqa: PLC0415 - lazy by design: Tier-1 import safety (module docstring)
    except ImportError:  # pragma: no cover - torch is a hard project dep
        return done_idx

    def _find_tensor(node: Any) -> torch.Tensor | None:  # noqa: ANN401 - recursive obs-tree walk
        if isinstance(node, torch.Tensor):
            return node
        if isinstance(node, dict):
            for child in node.values():
                hit = _find_tensor(child)
                if hit is not None:
                    return hit
        return None

    probe = _find_tensor(obs)
    if probe is None:
        return done_idx
    return torch.as_tensor(done_idx, dtype=torch.int32, device=probe.device)
