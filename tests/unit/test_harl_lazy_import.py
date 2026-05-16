# SPDX-License-Identifier: Apache-2.0
"""HARL lazy-import contract (ADR-002 §Revision-history 2026-05-16; #131).

PyPI rejects wheel ``METADATA`` containing ``git+URL`` direct references
(``400 Can't have direct dependency`` from the legacy upload endpoint),
so HARL is scoped to the ``[dependency-groups].train`` PEP 735 group in
``pyproject.toml`` rather than ``[project].dependencies``. Dependency
groups are uv-only and are *not* shipped in wheel METADATA, which lets
``concerto-multirobot`` upload cleanly while source-checkout users get
HARL via ``uv sync --group train``. The Phase-1 follow-up that publishes
``harl-aht`` to PyPI and restores a single-command training install is
tracked in https://github.com/fsafaei/concerto/issues/132.

The price of the train-group scoping is that pip-only consumers of the
safety stack and benchmark wrappers do *not* get HARL on
``pip install concerto-multirobot``. These tests pin the contract that
guards that surface:

- ``import chamber.benchmarks.ego_ppo_trainer`` and
  ``import chamber.partners.frozen_harl`` must succeed even when HARL
  is uninstalled — i.e. no module-level ``import harl`` may leak back
  in via a future code-style "tidy".
- Constructing :class:`~chamber.benchmarks.ego_ppo_trainer.EgoPPOTrainer`
  or loading a :class:`~chamber.partners.frozen_harl.FrozenHARLPartner`
  checkpoint without HARL must raise a :class:`ModuleNotFoundError`
  whose message names the install command (``uv sync --group train``)
  and the Phase-1 tracking issue, so users hitting it in the wild are
  pointed at the fix rather than at a bare ``No module named 'harl'``.

Implementation note: the test environment has the ``train`` group
installed (``make install`` resolves it), so we simulate "HARL
uninstalled" by replacing :func:`builtins.__import__` with a wrapper
that rejects any name in the ``harl.*`` namespace. Two fixtures split
the side effects: :func:`block_harl_imports` installs the blocker and
evicts cached ``harl.*`` entries from :data:`sys.modules`;
:func:`fresh_chamber_module_load` additionally evicts the lazy-import
chamber modules and clears their partner-registry entries so they can
re-run their top-level code without colliding on re-registration.
"""

from __future__ import annotations

import builtins
import importlib
import sys
from typing import Any

import pytest

#: Substring the helpful ``ModuleNotFoundError`` must contain. The install
#: command is the user-facing fix; pinning it here means a future code-
#: style change that silently demotes the message to ``No module named
#: 'harl'`` fails this test.
_INSTALL_HINT: str = "uv sync --group train"

#: The Phase-1 follow-up issue. Pinning the literal URL keeps the
#: assertion robust against wording shifts in the surrounding sentence.
_PHASE1_ISSUE_HINT: str = "https://github.com/fsafaei/concerto/issues/132"


@pytest.fixture
def block_harl_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every ``import harl...`` raise :class:`ModuleNotFoundError`.

    Two layers of defence:

    1. Evict any already-cached ``harl.*`` entries from :data:`sys.modules`
       so the trainer / partner constructors actually re-enter the import
       machinery rather than picking up a stale reference.
    2. Replace :func:`builtins.__import__` with a wrapper that rejects
       any name in the ``harl`` or ``harl.*`` namespace. All other
       imports pass through to the real ``__import__`` unchanged.

    Does *not* touch the chamber modules; use
    :func:`fresh_chamber_module_load` if the test needs a re-import of
    the lazy-import sites themselves.
    """
    real_import = builtins.__import__

    def blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "harl" or name.startswith("harl."):
            msg = f"No module named {name!r}"
            raise ModuleNotFoundError(msg)
        return real_import(name, *args, **kwargs)

    cached = [m for m in sys.modules if m == "harl" or m.startswith("harl.")]
    for mod_name in cached:
        monkeypatch.delitem(sys.modules, mod_name, raising=False)
    monkeypatch.setattr(builtins, "__import__", blocked_import)


@pytest.fixture
def fresh_chamber_module_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Force a fresh top-level load of the chamber lazy-import sites.

    Evicts ``chamber.benchmarks.ego_ppo_trainer`` and
    ``chamber.partners.frozen_harl`` from :data:`sys.modules`, and
    clears their :data:`chamber.partners.registry._REGISTRY` entries so
    the ``@register_partner("frozen_harl")`` decorator does not error
    on the re-import. Both side effects are restored at fixture
    teardown via :class:`pytest.MonkeyPatch`.
    """
    from chamber.partners import registry

    for mod_name in (
        "chamber.benchmarks.ego_ppo_trainer",
        "chamber.partners.frozen_harl",
    ):
        monkeypatch.delitem(sys.modules, mod_name, raising=False)
    # Clear the registry entry that ``chamber.partners.frozen_harl``
    # installs at import time — its decorator refuses to re-register
    # the same name. ``monkeypatch.delitem`` snapshots and restores the
    # entry at teardown so other tests in the session see the original
    # registration. Pyright marks ``_REGISTRY`` as private; the access
    # here is the canonical way to evict a registry entry for a
    # re-import test, so the access is intentional.
    monkeypatch.delitem(
        registry._REGISTRY,  # pyright: ignore[reportPrivateUsage]
        "frozen_harl",
        raising=False,
    )


@pytest.mark.usefixtures("block_harl_imports", "fresh_chamber_module_load")
class TestModuleImportContract:
    """The module top-level must not touch ``harl.*``.

    Verified by evicting the chamber lazy-import modules from
    :data:`sys.modules` (so the next ``import`` statement re-runs their
    top-level code) under the HARL import blocker.
    """

    def test_ego_ppo_trainer_module_imports_without_harl(self) -> None:
        """`import chamber.benchmarks.ego_ppo_trainer` must succeed without HARL.

        Regression guard against any future code-style cleanup that
        moves ``from harl... import ...`` back to module top.
        """
        mod = importlib.import_module("chamber.benchmarks.ego_ppo_trainer")
        assert hasattr(mod, "EgoPPOTrainer")

    def test_frozen_harl_module_imports_without_harl(self) -> None:
        """`import chamber.partners.frozen_harl` must succeed without HARL.

        Same regression guard for the partner-side module.
        :class:`chamber.partners.registry.load_partner` must remain a
        cheap registry lookup; only :meth:`FrozenHARLPartner._ensure_loaded`
        may trigger the HARL import.
        """
        mod = importlib.import_module("chamber.partners.frozen_harl")
        assert hasattr(mod, "FrozenHARLPartner")


@pytest.mark.usefixtures("block_harl_imports")
class TestRuntimeErrorContract:
    """The runtime import sites must raise a helpful error when HARL is missing.

    Does *not* evict the chamber modules; the already-loaded classes
    are used directly. The HARL imports live inside method bodies, so
    each constructor / loader call re-enters the import machinery and
    hits the blocker.
    """

    def test_ego_ppo_trainer_construction_names_install_command(self) -> None:
        """Constructing the trainer without HARL raises with the install hint.

        The construction path is :meth:`EgoPPOTrainer.from_config`,
        which builds the partner / env fixtures and then enters
        :meth:`EgoPPOTrainer.__init__`. The HARL import sits inside
        ``__init__``, after the partner-freeze gate (ADR-009
        §Consequences) and the seeding setup, so a frozen partner
        fixture is enough to reach the HARL import.
        """
        from chamber.benchmarks.ego_ppo_trainer import EgoPPOTrainer
        from chamber.envs.mpe_cooperative_push import MPECooperativePushEnv
        from chamber.partners.api import PartnerSpec
        from chamber.partners.heuristic import ScriptedHeuristicPartner
        from concerto.training.config import (
            EgoAHTConfig,
            EnvConfig,
            HAPPOHyperparams,
            PartnerConfig,
            RuntimeConfig,
        )

        cfg = EgoAHTConfig(
            seed=0,
            total_frames=16,
            checkpoint_every=16,
            env=EnvConfig(task="mpe_cooperative_push", episode_length=16),
            partner=PartnerConfig(
                class_name="scripted_heuristic",
                extra={
                    "uid": "partner",
                    "target_xy": "0.0,0.0",
                    "action_dim": "2",
                },
            ),
            happo=HAPPOHyperparams(
                rollout_length=16,
                batch_size=16,
                n_epochs=1,
                hidden_dim=32,
            ),
            runtime=RuntimeConfig(device="cpu"),
        )
        partner = ScriptedHeuristicPartner(
            spec=PartnerSpec(
                class_name="scripted_heuristic",
                seed=0,
                checkpoint_step=None,
                weights_uri=None,
                extra={
                    "uid": "partner",
                    "target_xy": "0.0,0.0",
                    "action_dim": "2",
                },
            )
        )
        env = MPECooperativePushEnv(root_seed=0)
        with pytest.raises(ModuleNotFoundError) as exc_info:
            EgoPPOTrainer.from_config(cfg, env=env, partner=partner, ego_uid="ego")
        assert _INSTALL_HINT in str(exc_info.value), (
            f"error message must name {_INSTALL_HINT!r}; got: {exc_info.value!s}"
        )
        assert _PHASE1_ISSUE_HINT in str(exc_info.value), (
            f"error message must name the Phase-1 tracking issue "
            f"({_PHASE1_ISSUE_HINT}); got: {exc_info.value!s}"
        )

    def test_frozen_harl_partner_load_names_install_command(self, tmp_path: Any) -> None:
        """Loading a :class:`FrozenHARLPartner` checkpoint without HARL raises with the hint.

        :meth:`FrozenHARLPartner._ensure_loaded` is the entry point —
        called on the first :meth:`reset` or :meth:`act`. The fixture's
        ``weights_uri`` does not need to point at a real checkpoint;
        the HARL import gate is checked *before* the checkpoint load,
        so an invalid path here would only surface if the HARL import
        succeeded (which it cannot, under the fixture).
        """
        from chamber.partners.api import PartnerSpec
        from chamber.partners.frozen_harl import FrozenHARLPartner

        partner = FrozenHARLPartner(
            spec=PartnerSpec(
                class_name="frozen_harl",
                seed=0,
                checkpoint_step=None,
                weights_uri=str(tmp_path / "nonexistent.pt"),
                extra={"uid": "partner"},
            )
        )
        with pytest.raises(ModuleNotFoundError) as exc_info:
            partner._ensure_loaded()  # pyright: ignore[reportPrivateUsage]
        assert _INSTALL_HINT in str(exc_info.value), (
            f"error message must name {_INSTALL_HINT!r}; got: {exc_info.value!s}"
        )
        assert _PHASE1_ISSUE_HINT in str(exc_info.value), (
            f"error message must name the Phase-1 tracking issue "
            f"({_PHASE1_ISSUE_HINT}); got: {exc_info.value!s}"
        )
