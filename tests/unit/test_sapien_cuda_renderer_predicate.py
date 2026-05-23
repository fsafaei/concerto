# SPDX-License-Identifier: Apache-2.0
"""Tier-1 tests for :func:`chamber.utils.device.sapien_cuda_renderer_available` (#188).

Pins the discriminator that the looser :func:`sapien_gpu_available`
gate could not enforce: ``test_draft_zoo_round_trip_on_real_stage0_env``
needs SAPIEN's *CUDA* render device to be functional (the
``panda_wristcam`` probe carries ``rgb`` + ``depth`` channels), and
the looser predicate returns True even when ManiSkill silently falls
back to the CPU renderer due to a host nvidia driver/library version
mismatch.

The tests stub :mod:`sapien` so they run on any host regardless of
the actual CUDA state — they pin the wrapper's behaviour, not the
host's hardware.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Any

import pytest

from chamber.utils import device as device_module


def _install_fake_sapien(
    monkeypatch: pytest.MonkeyPatch, *, device_raises: Exception | None
) -> None:
    """Stub :mod:`sapien` with a :class:`Device` that optionally raises.

    Args:
        device_raises: If not None, ``sapien.Device("cuda")`` will raise
            this exception (the failure mode the predicate is designed
            to catch). If None, the call succeeds and returns a sentinel.
    """
    fake = ModuleType("sapien")

    class _Device:
        def __init__(self, name: str) -> None:
            del name
            if device_raises is not None:
                raise device_raises

    fake.Device = _Device  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sapien", fake)
    # Force the predicate to re-import sapien through the stub.
    importlib.reload(device_module)


class TestSapienCudaRendererAvailable:
    """Pin the #188 predicate contract."""

    def test_returns_true_when_cuda_device_construction_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Happy path: ``sapien.Device('cuda')`` resolves without raising."""
        _install_fake_sapien(monkeypatch, device_raises=None)
        from chamber.utils.device import sapien_cuda_renderer_available as fn

        assert fn() is True

    def test_returns_false_on_cuda_device_not_found_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The exact failure mode ManiSkill's backend resolver catches.

        ``mani_skill.envs.utils.system.backend`` catches
        ``RuntimeError: failed to find device "cuda"`` and silently
        falls back to the CPU renderer. The new predicate must surface
        that exact case as False (the whole point of #188).
        """
        _install_fake_sapien(
            monkeypatch,
            device_raises=RuntimeError('failed to find device "cuda"'),
        )
        from chamber.utils.device import sapien_cuda_renderer_available as fn

        assert fn() is False

    def test_returns_false_on_arbitrary_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The predicate swallows every exception type (parallel to ``sapien_gpu_available``)."""
        _install_fake_sapien(monkeypatch, device_raises=ValueError("boom"))
        from chamber.utils.device import sapien_cuda_renderer_available as fn

        assert fn() is False

    def test_returns_false_when_sapien_is_not_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ImportError path: no SAPIEN on the host (CPU-only contributor laptop)."""
        # Block ``import sapien`` even if it happens to be installed.
        monkeypatch.setitem(sys.modules, "sapien", None)
        importlib.reload(device_module)
        from chamber.utils.device import sapien_cuda_renderer_available as fn

        assert fn() is False

    def test_predicate_is_pure_query_no_side_effects(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ADR-001 §Risks: device helpers must be safe to call at import time.

        Calling the predicate twice must produce the same result and not
        mutate any global state observable to the caller.
        """
        _install_fake_sapien(monkeypatch, device_raises=None)
        from chamber.utils.device import sapien_cuda_renderer_available as fn

        first = fn()
        second = fn()
        assert first is second is True


class TestDeviceReportIncludesCudaRendererStatus:
    """Pin the device_report widening (#188): operators must see the tight gate."""

    def test_report_carries_cuda_renderer_field(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``device_report()`` must mention SAPIEN/CUDA renderer status alongside SAPIEN/Vulkan."""
        _install_fake_sapien(monkeypatch, device_raises=None)
        from chamber.utils.device import device_report

        report = device_report()
        assert "SAPIEN/CUDA renderer" in report
        assert "SAPIEN/Vulkan" in report
        assert "PyTorch device" in report

    def test_report_reflects_cpu_fallback_state(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When CUDA-renderer probe fails, the report must say ``unavailable``."""
        _install_fake_sapien(
            monkeypatch,
            device_raises=RuntimeError('failed to find device "cuda"'),
        )
        from chamber.utils.device import device_report

        report = device_report()
        assert "SAPIEN/CUDA renderer: unavailable" in report


@pytest.fixture(autouse=True)
def _restore_device_module() -> Any:
    """Reload ``chamber.utils.device`` post-test so a stubbed sapien doesn't leak.

    The tests install a fake ``sapien`` module via ``monkeypatch.setitem``;
    pytest's monkeypatch restores the original ``sys.modules`` entry on
    teardown, but the cached re-import of ``chamber.utils.device`` we
    triggered via ``importlib.reload`` would still hold references to
    the fake module's classes. Reload the device module once more after
    each test so the next test (or downstream test in the same session)
    gets a clean view.
    """
    yield
    importlib.reload(device_module)
