# SPDX-License-Identifier: Apache-2.0
"""Task registry — the single source of truth for CHAMBER-Bench (ADR-027 §Versioning).

Mirrors the ``chamber.partners`` plug-in registry (ADR-009 §Decision):
a module-global table, a ``@register_task`` decorator that loud-fails
on duplicates (``ValueError``), and lookups that loud-fail on unknown
ids (``KeyError`` listing the known keys). The registry is populated at
import time by :mod:`chamber.tasks.ladder`; everything downstream — the
generated manifest, the README task table, the ``docs/reference/tasks/``
cards — renders from it, so the suite composition cannot drift from
code (ADR-027 §Consequences).

Env construction stays lazy: :func:`make` resolves the spec's dotted
``env_factory`` path only when called, so importing :mod:`chamber.tasks`
is Tier-1-safe on a Vulkan-less host (ADR-001 §Risks / P2).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from chamber.tasks.spec import TaskSpec

#: Suite identity pinned by the generated manifest (ADR-027 §Versioning).
SUITE_NAME: str = "CHAMBER-Bench"
SUITE_VERSION: str = "1.0"

# task_id -> {version -> TaskSpec}
_REGISTRY: dict[str, dict[int, TaskSpec]] = {}


def register_task(builder: Callable[[], TaskSpec]) -> Callable[[], TaskSpec]:
    """Register the :class:`TaskSpec` returned by ``builder`` (ADR-027 §Versioning).

    Decorator over a zero-argument builder function; the spec is built
    and registered eagerly at decoration time. Re-registering an
    existing ``task_id@version`` raises ``ValueError`` — a changed task
    must bump its version, never overwrite a registered spec (ADR-027
    §Versioning; §Reversibility).
    """
    spec = builder()
    versions = _REGISTRY.setdefault(spec.task_id, {})
    if spec.version in versions:
        msg = f"task {spec.slug!r} is already registered"
        raise ValueError(msg)
    versions[spec.version] = spec
    return builder


def get(task_id: str, version: int | None = None) -> TaskSpec:
    """Return the :class:`TaskSpec` for ``task_id`` (ADR-027 §Versioning).

    ``version=None`` (the default) resolves to the latest registered
    version. Unknown ids and unknown versions raise ``KeyError``
    listing the known keys (the ADR-009 §Decision registry error
    style).
    """
    versions = _REGISTRY.get(task_id)
    if versions is None:
        known = ", ".join(sorted(_REGISTRY)) or "<none>"
        msg = f"unknown task id {task_id!r}; registered task ids: {known}"
        raise KeyError(msg)
    if version is None:
        version = max(versions)
    if version not in versions:
        known_versions = ", ".join(str(v) for v in sorted(versions))
        msg = (
            f"unknown version {version} for task {task_id!r}; registered versions: {known_versions}"
        )
        raise KeyError(msg)
    return versions[version]


def list_registered() -> list[str]:
    """Sorted ``task_id@vN`` slugs of every registered task version (ADR-027 §Versioning)."""
    return sorted(spec.slug for versions in _REGISTRY.values() for spec in versions.values())


def make(task_id: str, *, version: int | None = None, **overrides: Any) -> Any:  # noqa: ANN401 - delegates verbatim to per-task env factories with heterogeneous signatures/returns
    """Construct the task's env by delegating to its registered factory (ADR-027 §Versioning).

    Resolves the spec's dotted ``env_factory`` path lazily (Tier-1
    contract, ADR-001 §Risks / P2) and calls it with the spec's
    ``factory_defaults`` merged under the caller's ``overrides``. The
    registry adds **no env behaviour** — it is pure delegation to the
    existing ``chamber.envs`` / ``chamber.benchmarks`` factories (P2
    wrapper-only rule, ADR-001 §Decision).

    Spec-only placeholders (``env_factory is None``) raise
    ``NotImplementedError`` with a pointer at the task card.
    """
    spec = get(task_id, version=version)
    if spec.env_factory is None:
        msg = (
            f"{spec.slug} is a spec-only Tier-{spec.tier} placeholder with no env "
            f"factory wired; it documents a candidate task (ADR-027 §Tier ladder). "
            f"See docs/reference/tasks/{spec.task_id}.md for its committed scope."
        )
        raise NotImplementedError(msg)
    module_name, _, attr = spec.env_factory.rpartition(".")
    factory = getattr(importlib.import_module(module_name), attr)
    kwargs: dict[str, Any] = {**spec.factory_defaults, **overrides}
    return factory(**kwargs)


def manifest() -> dict[str, Any]:
    """Return the pinned CHAMBER-Bench suite composition (ADR-027 §Versioning).

    Deterministically ordered — tasks sorted by ``(tier, task_id,
    version)``, fields in :class:`TaskSpec` declaration order — so two
    renders are byte-identical and the manifest can be diffed as a
    drift gate (ADR-027 §Validation criteria: "the manifest generator
    pins the suite").
    """
    specs = sorted(
        (spec for versions in _REGISTRY.values() for spec in versions.values()),
        key=lambda s: (s.tier, s.task_id, s.version),
    )
    return {
        "suite": SUITE_NAME,
        "suite_version": SUITE_VERSION,
        "tasks": [spec.model_dump(mode="json") for spec in specs],
    }


__all__ = [
    "SUITE_NAME",
    "SUITE_VERSION",
    "get",
    "list_registered",
    "make",
    "manifest",
    "register_task",
]
