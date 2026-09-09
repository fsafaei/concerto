# SPDX-License-Identifier: Apache-2.0
"""Tier-1 tests for the Gymnasium ids ``concerto.contracts.sim`` registers (no env is built)."""

from __future__ import annotations

import importlib
import inspect
import re
import warnings

import gymnasium as gym

from concerto import contracts


def _sim():
    return importlib.import_module("concerto.contracts.sim")


def test_every_contract_version_has_a_registered_gym_id() -> None:
    _sim()
    for task_id in contracts.list_contracts():
        for version in contracts.versions(task_id):
            c = contracts.get(task_id, version)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)  # older versions stay usable
                spec = gym.spec(c.gym_id)
            assert spec.entry_point == "concerto.contracts.sim:make_gym_env"
            assert spec.kwargs == {"task_id": task_id, "version": version}
            assert spec.disable_env_checker is True
            assert spec.max_episode_steps is None, "the inner make applies the contract's horizon"


def test_register_gym_ids_is_idempotent_and_quiet() -> None:
    sim = _sim()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert sim.register_gym_ids() == []


def test_sim_module_never_reads_env_spec() -> None:
    # gymnasium caches ``Wrapper.spec`` on first access and the outer ``gym.make`` rewrites the
    # unwrapped spec only after the entry point returns: reading it inside the factories would
    # freeze the inner ManiSkill id and hide the contract's gym id.
    src = inspect.getsource(_sim())
    assert re.search(r"\benv\.spec\b|\.spec\.(id|max_episode_steps)", src) is None
