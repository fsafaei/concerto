# SPDX-License-Identifier: Apache-2.0
"""Ego-only HAPPO with frozen partner — CONCERTO ADR-002 risk-mitigation #1.

Drop-in subclass of upstream HARL's :class:`HAPPO` that adapts the
sequential-update scheme to the AHT setting:

- the partner's parameters are NOT collected for the joint update;
- the partner produces actions through a frozen forward pass only
  (``torch.no_grad``);
- the per-step advantage is computed for the ego only;
- the sequential update reduces to a single-agent PPO update.

This loses Theorem 7's formal monotonic-improvement guarantee (which
requires *all* agents to update simultaneously). CONCERTO verifies
empirically (T4b.13) that per-epoch reward is non-decreasing on a
2-agent MPE Cooperative-Push task on >=80% of moving-window-of-10
intervals; if the assertion fails, the ADR-002 risk-mitigation #1
trigger fires and Phase-0 stops to revisit the framework choice.

This module lives inside the ``concerto-org/harl-fork`` repo at
``harl/algorithms/actors/ego_aht_happo.py``. The patch is generated
from CONCERTO's
``scripts/harl-fork-patches/v0.1.0-aht/`` directory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

# UPSTREAM-VERIFY: the canonical upstream import path is
# ``harl.algorithms.actors.happo.HAPPO``. Verify against the pinned
# ``v0.0.0-vendored`` commit; if upstream HARL has reorganised the
# actor module path, adjust here.
from harl.algorithms.actors.happo import HAPPO

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray


class EgoAHTHAPPO(HAPPO):
    """Drop-in HAPPO subclass for AHT frozen-partner training (CONCERTO ADR-002 risk #1).

    The partner argument MUST be a frozen :mod:`torch.nn.Module`-bearing
    object (every parameter has ``requires_grad == False``).
    Construction raises :class:`ValueError` if any parameter is trainable
    — the ADR-009 §Consequences runtime backstop for the AHT
    no-joint-training contract.

    The ``partner_adapter`` argument satisfies the
    :class:`concerto.training.ego_aht.PartnerLike` Protocol structurally:
    it has ``act(obs, deterministic=True) -> NDArray[np.floating]``,
    ``reset(seed)`` and a ``spec`` attribute. The fork does not import
    that Protocol explicitly; the CONCERTO-side runner type-checks it
    when it constructs the partner.
    """

    def __init__(
        self,
        *args: Any,
        partner_adapter: Any,
        **kwargs: Any,
    ) -> None:
        """Construct the wrapper + validate the frozen-partner contract.

        Args:
            *args: Forwarded to :class:`HAPPO`'s upstream constructor.
                # UPSTREAM-VERIFY: HARL's ``HAPPO.__init__`` accepts
                # ``(args, obs_space, act_space, device)`` per the
                # commit-of-record; verify against the pinned v0.0.0
                # commit and adjust this docstring.
            partner_adapter: Frozen partner satisfying the
                :class:`concerto.training.ego_aht.PartnerLike` Protocol.
                Construction validates that every torch parameter is
                frozen.
            **kwargs: Forwarded to :class:`HAPPO`'s upstream constructor.

        Raises:
            ValueError: If any parameter on any submodule of
                ``partner_adapter`` has ``requires_grad == True``.
        """
        super().__init__(*args, **kwargs)
        self._partner = partner_adapter
        self._validate_partner_is_frozen()

    def _validate_partner_is_frozen(self) -> None:
        """Refuse a partner with any trainable parameter (ADR-009 §Consequences).

        Walks every :class:`torch.nn.Module` reachable from the partner
        adapter and asserts ``requires_grad == False`` on every named
        parameter. The ego training loop must NOT propagate gradients
        through the partner — this is the runtime backstop for the
        black-box AHT contract.

        Raises:
            ValueError: If any parameter has ``requires_grad == True``.
        """
        for module in self._iter_partner_modules():
            for name, param in module.named_parameters():
                if param.requires_grad:
                    raise ValueError(
                        f"Partner adapter has trainable parameter {name!r} "
                        "with requires_grad=True. ADR-009 §Consequences: "
                        "the AHT frozen-partner contract requires every "
                        "partner parameter to have requires_grad=False. "
                        "Freeze the partner via "
                        "`for p in partner.parameters(): p.requires_grad = False` "
                        "before passing it to EgoAHTHAPPO."
                    )

    def _iter_partner_modules(self) -> list[nn.Module]:
        """Enumerate :class:`nn.Module` instances reachable from the partner.

        Phase-0 partners come in two shapes: scripted (no ``nn.Module`` —
        the heuristic carries no parameters) and frozen-RL adapters
        (carry one or more ``nn.Module`` heads). The check is a no-op
        for the scripted case (no modules → no params → no validation
        failures) and exhaustive for the frozen-RL case.

        Limitation: the walk inspects only attributes that are
        ``nn.Module`` directly, not modules nested inside lists or
        dicts (e.g. ``self._partner.heads = [nn.Linear(...), ...]``
        would not be detected). Phase-0 partners do not nest modules
        in collections; if Phase-1 adds a partner that does, extend
        this method to walk container attributes.
        """
        if isinstance(self._partner, nn.Module):
            return list(self._partner.modules())
        # Heuristic partners may not be nn.Modules; walk attributes for
        # the rare case of a partner that nests a torch model under e.g.
        # ``self._partner.policy``.
        modules: list[nn.Module] = []
        for attr in vars(self._partner).values():
            if isinstance(attr, nn.Module):
                modules.extend(attr.modules())
        return modules

    def collect_rollout(self, env: Any, n_steps: int) -> Any:
        """Collect ``n_steps`` of rollout with the ego learning + partner frozen.

        # UPSTREAM-VERIFY: HARL's ``HAPPO.collect_rollout`` (or the
        # equivalent method — some HARL versions name it ``run`` and
        # delegate to a ``Runner`` class) is the canonical place to
        # interpose the partner's frozen forward pass. Confirm the
        # method name + signature on the pinned ``v0.0.0-vendored``
        # commit and adapt the override in place. The pattern is:
        #
        #   for step in range(n_steps):
        #       ego_action = super().act(obs)               # learnable
        #       with torch.no_grad():
        #           partner_action = self._partner.act(obs, deterministic=True)
        #       action = {ego_uid: ego_action, partner_uid: partner_action}
        #       obs, reward, done, info = env.step(action)
        #       self._buffer.add(obs, ego_action, reward, done, info)
        #
        # If your HARL version uses a Runner-based loop, move this body
        # into :class:`harl.runners.ego_aht_runner.EgoAHTRunner.collect`
        # and have ``collect_rollout`` delegate.
        """
        del env, n_steps
        raise NotImplementedError(
            "Verify upstream HAPPO.collect_rollout signature against the "
            "pinned v0.0.0-vendored commit; this method intentionally "
            "raises until that wiring is done. See the # UPSTREAM-VERIFY: "
            "comment above for the expected pattern."
        )

    def update(self, buffer: Any) -> Any:
        """Compute ego-only advantage + delegate to single-agent PPO update.

        Phase-0 simplification: the partner's actions are treated as
        fixed environmental dynamics. The per-step advantage is the
        ego's PPO advantage; the partner contributes no policy gradient.

        # UPSTREAM-VERIFY: HARL's ``HAPPO.update`` may iterate over all
        # agents in a sequential-update loop. For the AHT case the loop
        # collapses to ``len(agents) == 1``: only the ego updates. The
        # simplest implementation is to delegate to ``super().update``
        # with the buffer narrowed to the ego's trajectories; if the
        # upstream signature does not allow this, override the
        # advantage-computation method instead.
        #
        # NOTE: this method intentionally raises until the buffer-
        # narrowing is verified against the pinned upstream commit. A
        # silent ``return super().update(buffer)`` could otherwise
        # train on the full multi-agent buffer (including partner
        # trajectories), which would violate the AHT no-joint-training
        # contract (ADR-009 §Consequences).
        """
        del buffer
        raise NotImplementedError(
            "Verify upstream HAPPO.update buffer signature against the "
            "pinned v0.0.0-vendored commit and narrow the buffer to ego "
            "trajectories before delegating. See the # UPSTREAM-VERIFY: "
            "comment above."
        )

    @classmethod
    def from_config(
        cls,
        cfg: Any,
        *,
        env: Any,
        ego_uid: str,
    ) -> EgoAHTHAPPO:
        """Construct from a CONCERTO :class:`EgoAHTConfig` (M4b-7 wires this up).

        Bridges :class:`concerto.training.config.EgoAHTConfig` to HARL's
        expected constructor args. This classmethod is the
        :class:`concerto.training.ego_aht.TrainerFactory` callable that
        :func:`concerto.training.ego_aht.train` plugs in via dependency
        injection.

        Args:
            cfg: Validated :class:`concerto.training.config.EgoAHTConfig`.
            env: Concrete env (Gymnasium-multi-agent shape).
            ego_uid: The env-side uid the ego acts on.

        Returns:
            A constructed :class:`EgoAHTHAPPO` ready for the loop.

        # UPSTREAM-VERIFY: HARL's ``HAPPO.__init__`` arg list is the
        # blocker here. Map cfg.happo (lr / gamma / gae_lambda /
        # clip_eps / n_epochs / rollout_length / batch_size /
        # hidden_dim) to the matching HAPPO kwargs once you have the
        # signature in front of you. The partner_adapter argument is
        # constructed by the chamber-side runner — see
        # ``chamber.benchmarks.training_runner.build_partner``.
        """
        del cfg, env, ego_uid
        raise NotImplementedError(
            "M4b-7: wire this once the upstream HAPPO constructor "
            "signature is verified against the pinned v0.0.0-vendored "
            "commit. The mapping from cfg.happo.* to HAPPO kwargs is "
            "the only non-trivial piece."
        )


__all__ = ["EgoAHTHAPPO"]
