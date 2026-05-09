# SPDX-License-Identifier: Apache-2.0
"""Adapter from a CONCERTO Gymnasium-multi-agent env to HARL's runner API.

Lives inside the ``concerto-org/harl-fork`` repo at
``harl/envs/concerto_env_adapter.py``. Generated from CONCERTO's
``scripts/harl-fork-patches/v0.1.0-aht/`` recipe.

CONCERTO's env classes (:class:`chamber.envs.mpe_cooperative_push.MPECooperativePushEnv`
and the M4b-7 Stage-0 wrapper) are Gymnasium-compatible:

- ``reset(*, seed, options) -> (obs, info)``
- ``step(action: dict[uid, action]) -> (obs, reward, terminated, truncated, info)``
- ``action_space`` / ``observation_space`` exposed as :class:`gymnasium.spaces.Dict`.

Upstream HARL's runner expects a slightly different interface. This
adapter is the thin shim — no new state, no new dispatch logic — that
lets HARL consume the CONCERTO env unchanged.

# UPSTREAM-VERIFY: HARL's expected env protocol varies across versions.
# Verify against the pinned ``v0.0.0-vendored`` commit:
# - Some HARL releases require ``env.reset()`` to return only ``obs``
#   (Gym pre-0.26 style); newer HARL aligns with Gymnasium's tuple
#   return.
# - HARL's ``env.step`` may expect ``(obs, reward, done, info)`` (Gym
#   pre-0.26) instead of Gymnasium's 5-tuple.
# Adjust the adapter if upstream HARL is on the older API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping


class ConcertoEnvAdapter:
    """Wrap a CONCERTO env so HARL's runner can use it (CONCERTO T4b.3).

    The adapter is a structural pass-through: every method delegates
    to the underlying env. Phase-0 ships only the methods the HARL
    runner actually calls; if upstream HARL grows new requirements,
    add them here rather than patching the inner env.
    """

    def __init__(self, env: Any) -> None:
        """Bind the inner CONCERTO env (T4b.3).

        Args:
            env: A CONCERTO env satisfying the
                :class:`concerto.training.ego_aht.EnvLike` Protocol
                (e.g. :class:`MPECooperativePushEnv` or the M4b-7
                Stage-0 wrapper).
        """
        self._env = env

    def reset(self, **kwargs: Any) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        """Delegate to the inner env's ``reset`` (T4b.3).

        Returns:
            ``(obs, info)`` — the Gymnasium 2-tuple. If your pinned HARL
            expects only ``obs``, narrow the return at the call site
            inside :class:`harl.runners.ego_aht_runner.main`.
        """
        return self._env.reset(**kwargs)

    def step(
        self,
        action: dict[str, Any],
    ) -> tuple[Mapping[str, Any], float, bool, bool, Mapping[str, Any]]:
        """Delegate to the inner env's ``step`` (T4b.3).

        Args:
            action: Dict-keyed multi-agent action.

        Returns:
            ``(obs, reward, terminated, truncated, info)`` — the
            Gymnasium 5-tuple. Same caveat as :meth:`reset` if upstream
            HARL is on the older 4-tuple API.
        """
        return self._env.step(action)

    @property
    def action_space(self) -> Any:
        """Expose the inner env's action space (T4b.3)."""
        return self._env.action_space

    @property
    def observation_space(self) -> Any:
        """Expose the inner env's observation space (T4b.3)."""
        return self._env.observation_space

    def close(self) -> None:
        """No-op close (Gymnasium contract; ADR-002 §Decisions).

        CONCERTO envs are pure-Python and hold no external resources.
        Defined here so HARL runners that call ``env.close()`` at end
        of run do not crash.
        """


__all__ = ["ConcertoEnvAdapter"]
