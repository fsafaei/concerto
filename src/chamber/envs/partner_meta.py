# SPDX-License-Identifier: Apache-2.0
"""Partner-identity annotation wrapper (ADR-006 risk #3; ADR-004 §risk-mitigation #2).

Surfaces ``obs["meta"]["partner_id"]`` on every :meth:`reset` and
:meth:`step` so the M3 conformal filter (and any downstream consumer)
can detect a mid-episode partner swap and re-initialise the conformal
``lambda`` to ``lambda_safe`` per ADR-004 §risk-mitigation #2. The
:class:`~chamber.partners.api.PartnerSpec.partner_id` attribute is the
stable hash this wrapper writes — see
:attr:`chamber.partners.api.PartnerSpec.partner_id` for the hash
material contract (plan/04 §3.1).

The wrapper is intentionally separated from
:func:`chamber.benchmarks.stage0_smoke.make_stage0_env` (which knows
nothing about partners): consumers compose
``PartnerIdAnnotationWrapper(inner_env, partner_id=spec.partner_id)``
once they have built the partner. The Phase-0 single-partner-per-
episode case fixes ``partner_id`` at construction; the
:meth:`set_partner_id` setter exists so a Phase-1 Stage-3 PF spike with
mid-episode swap can re-bind the annotation before the next step
without rebuilding the wrapper.

The wrapper does not mutate any other obs key. ``obs["meta"]`` is built
on top of whatever meta dict the inner env may already provide — for
Phase-0 envs that don't carry a meta dict, this just creates one with
the single ``partner_id`` entry. Future meta keys (run identifiers,
seeds, etc.) compose by adding their own wrapper above this one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym

from chamber.envs.errors import ChamberEnvCompatibilityError

if TYPE_CHECKING:
    from collections.abc import Mapping

#: Top-level obs key under which the wrapper writes its meta dict.
#:
#: Matches the producer-side commitment named in
#: :class:`chamber.partners.api.PartnerSpec.partner_id`'s docstring
#: (plan/04 §3.1). Renaming this is a wire-format break that requires
#: an ADR-006 amendment.
_META_KEY: str = "meta"

#: Sub-key inside ``obs["meta"]`` that carries the partner identity.
#:
#: Matches the consumer contract in
#: :func:`concerto.safety.conformal.reset_on_partner_swap` (ADR-004
#: §risk-mitigation #2). Renaming is a wire-format break.
_PARTNER_ID_KEY: str = "partner_id"


class PartnerIdAnnotationWrapper(gym.Wrapper):  # type: ignore[type-arg]
    """Inject ``obs["meta"]["partner_id"]`` on every reset and step (ADR-006 risk #3).

    Wraps any env whose observation space is a :class:`gym.spaces.Dict`.
    The wrapper extends the obs space with a ``"meta"`` sub-Dict (empty
    in Phase-0 — Gymnasium's :class:`gym.spaces.Dict` does not natively
    type string values, so the actual ``partner_id`` field is carried
    in the runtime observation dict rather than declared in the space).

    Note:
        :meth:`gym.spaces.Dict.contains` returns ``False`` on the
        ``"meta"`` sub-dict because the declared inner space is empty;
        this matches :class:`chamber.envs.CommShapingWrapper`'s
        precedent for its ``"comm"`` sub-space (no current consumer
        validates an obs via ``observation_space.contains``). Adding a
        string-typed inner Space would require an upstream Gymnasium
        change.

    Phase-0 binds ``partner_id`` at construction. For a Phase-1
    Stage-3 PF spike with mid-episode swap, call :meth:`set_partner_id`
    before the next :meth:`step`; the wrapper will surface the new
    value on the subsequent obs.

    Args:
        env: Inner env with a :class:`gym.spaces.Dict` observation space.
        partner_id: Stable 16-hex-char hash from
            :attr:`chamber.partners.api.PartnerSpec.partner_id`. The
            wrapper does not validate the format — :class:`PartnerSpec`
            is the source of truth.

    Raises:
        ChamberEnvCompatibilityError: When the inner observation space
            is not a :class:`gym.spaces.Dict` (ADR-001 §Risks; matches
            :class:`chamber.envs.CommShapingWrapper`'s compatibility
            contract).
    """

    def __init__(self, env: gym.Env, *, partner_id: str) -> None:  # type: ignore[type-arg]
        """Validate the obs space and bind the partner identity (ADR-006 risk #3)."""
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Dict):
            msg = (
                f"PartnerIdAnnotationWrapper requires a gym.spaces.Dict observation "
                f"space; got {type(env.observation_space).__name__}. See ADR-001 §Risks."
            )
            raise ChamberEnvCompatibilityError(msg)
        self._partner_id: str = partner_id
        self._extend_observation_space()

    def _extend_observation_space(self) -> None:
        """Add an empty ``"meta"`` sub-space (Gymnasium has no string-typed Space)."""
        existing = dict(self.env.observation_space.spaces)  # type: ignore[attr-defined]
        existing.setdefault(_META_KEY, gym.spaces.Dict({}))
        self.observation_space = gym.spaces.Dict(existing)

    @property
    def partner_id(self) -> str:
        """Return the partner identity the wrapper currently injects (ADR-006 risk #3).

        Returns:
            The stable 16-hex-char hash bound at construction or via
            :meth:`set_partner_id`.
        """
        return self._partner_id

    def set_partner_id(self, partner_id: str) -> None:
        """Rebind the partner identity for the next reset / step (ADR-004 §risk-mitigation #2).

        Used by Phase-1 mid-episode-swap experiments (ADR-007 Stage-3 PF):
        call this before the next :meth:`step` and the wrapper will
        surface the new value on the subsequent obs. The change does
        NOT trigger a new :meth:`reset` automatically — the conformal
        filter does that itself when it observes the new
        ``partner_id`` on ``obs["meta"]``.

        Args:
            partner_id: New stable hash from
                :attr:`chamber.partners.api.PartnerSpec.partner_id`.
        """
        self._partner_id = partner_id

    def reset(  # type: ignore[override]
        self, **kwargs: object
    ) -> tuple[dict, dict]:
        """Reset the env and inject ``partner_id`` into the initial obs (ADR-006 risk #3)."""
        obs, info = self.env.reset(**kwargs)  # type: ignore[arg-type]
        return self._annotate(obs), info

    def step(  # type: ignore[override]
        self, action: object
    ) -> tuple[dict, object, bool, bool, dict]:
        """Step the env and inject the current ``partner_id`` (ADR-006 risk #3)."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._annotate(obs), reward, terminated, truncated, info

    def _annotate(self, obs: Mapping[str, object]) -> dict[str, object]:
        """Return a copy of ``obs`` with ``obs["meta"]["partner_id"]`` set."""
        out = dict(obs)
        existing_meta = out.get(_META_KEY)
        meta: dict[str, object] = dict(existing_meta) if isinstance(existing_meta, dict) else {}
        meta[_PARTNER_ID_KEY] = self._partner_id
        out[_META_KEY] = meta
        return out


__all__ = ["PartnerIdAnnotationWrapper"]
