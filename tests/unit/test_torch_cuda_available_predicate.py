# SPDX-License-Identifier: Apache-2.0
"""Tier-1 tests for :func:`chamber.utils.device.torch_cuda_available` (#198).

Pins the gate that the looser :func:`sapien_gpu_available` predicate
could not enforce: Tier-2 integration tests that construct an
:class:`EgoPPOTrainer` with ``RuntimeConfig.device='cuda'`` need to
skip cleanly on hosts where SAPIEN/Vulkan probes succeed but torch's
CUDA stack is broken (nvidia kernel-module / userspace NVML mismatch
pending a reboot; driver below the cu128 native-support minimum on
consumer hardware, per ADR-002 §Rev 2026-05-20 cuda-major coupling
discipline). Without the gate, the tests fail with
``RuntimeConfig.device='cuda' requested but torch.cuda.is_available()
is False`` from
:func:`chamber.benchmarks.ego_ppo_trainer._resolve_device` and bounce
the PR.

The tests stub :mod:`torch` so they run on any host regardless of the
actual CUDA state — they pin the wrapper's behaviour, not the host's
hardware.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Any

import pytest

from chamber.utils import device as device_module


def _install_fake_torch(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cuda_is_available: bool | Exception,
) -> None:
    """Stub :mod:`torch` so ``torch.cuda.is_available()`` returns / raises as told.

    Args:
        cuda_is_available: If a bool, ``torch.cuda.is_available()`` returns
            it. If an exception instance, the call raises that exception
            (mirrors the torch-internal failure modes the predicate must
            collapse to ``False``).
    """
    fake = ModuleType("torch")
    fake_cuda = ModuleType("torch.cuda")
    fake_backends = ModuleType("torch.backends")
    fake_mps = ModuleType("torch.backends.mps")

    def _cuda_is_available() -> bool:
        if isinstance(cuda_is_available, Exception):
            raise cuda_is_available
        return cuda_is_available

    fake_cuda.is_available = _cuda_is_available  # type: ignore[attr-defined]
    # ``torch_device()`` (called transitively by ``device_report()``) probes
    # MPS after CUDA; stub a False-returning ``mps.is_available`` so the
    # report path doesn't AttributeError on hosts that don't ship MPS.
    fake_mps.is_available = lambda: False  # type: ignore[attr-defined]
    fake_backends.mps = fake_mps  # type: ignore[attr-defined]
    fake.cuda = fake_cuda  # type: ignore[attr-defined]
    fake.backends = fake_backends  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setitem(sys.modules, "torch.cuda", fake_cuda)
    monkeypatch.setitem(sys.modules, "torch.backends", fake_backends)
    monkeypatch.setitem(sys.modules, "torch.backends.mps", fake_mps)
    importlib.reload(device_module)


class TestTorchCudaAvailable:
    """Pin the #198 predicate contract."""

    def test_returns_true_when_torch_cuda_is_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Happy path: ``torch.cuda.is_available()`` returns True."""
        _install_fake_torch(monkeypatch, cuda_is_available=True)
        from chamber.utils.device import torch_cuda_available as fn

        assert fn() is True

    def test_returns_false_when_torch_cuda_is_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CPU-only host: ``torch.cuda.is_available()`` returns False."""
        _install_fake_torch(monkeypatch, cuda_is_available=False)
        from chamber.utils.device import torch_cuda_available as fn

        assert fn() is False

    def test_returns_false_on_runtime_error_from_torch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The driver/library mismatch path that prompted #198.

        On a host where the loaded nvidia kernel module is out of sync
        with the userspace NVML library (e.g. after an unattended driver
        upgrade pending a reboot), ``torch.cuda.is_available()`` issues a
        ``UserWarning`` and may surface as an internal exception in some
        torch builds; the predicate must collapse the failure to
        ``False`` rather than propagate.
        """
        _install_fake_torch(
            monkeypatch,
            cuda_is_available=RuntimeError(
                "CUDA error 804 forward compatibility on non supported HW"
            ),
        )
        from chamber.utils.device import torch_cuda_available as fn

        assert fn() is False

    def test_returns_false_on_arbitrary_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The predicate swallows every exception type (parallel to ``sapien_gpu_available``)."""
        _install_fake_torch(monkeypatch, cuda_is_available=ValueError("boom"))
        from chamber.utils.device import torch_cuda_available as fn

        assert fn() is False

    def test_returns_false_when_torch_is_not_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ImportError path: no torch on the host (extremely unusual but defended)."""
        monkeypatch.setitem(sys.modules, "torch", None)
        importlib.reload(device_module)
        from chamber.utils.device import torch_cuda_available as fn

        assert fn() is False

    def test_predicate_is_pure_query_no_side_effects(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ADR-001 §Risks: device helpers must be safe to call at import time.

        Calling the predicate twice must produce the same result and not
        mutate any global state observable to the caller.
        """
        _install_fake_torch(monkeypatch, cuda_is_available=True)
        from chamber.utils.device import torch_cuda_available as fn

        first = fn()
        second = fn()
        assert first is second is True


class TestDeviceReportIncludesTorchCudaStatus:
    """Pin the device_report widening (#198): operators must see the torch.cuda gate."""

    def test_report_carries_torch_cuda_field(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``device_report()`` must mention torch.cuda alongside SAPIEN gates."""
        _install_fake_torch(monkeypatch, cuda_is_available=True)
        from chamber.utils.device import device_report

        report = device_report()
        assert "torch.cuda" in report
        assert "SAPIEN/Vulkan" in report
        assert "SAPIEN/CUDA renderer" in report
        assert "PyTorch device" in report

    def test_report_reflects_torch_cuda_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When the torch.cuda probe returns False, the report must say ``unavailable``."""
        _install_fake_torch(monkeypatch, cuda_is_available=False)
        from chamber.utils.device import device_report

        report = device_report()
        assert "torch.cuda: unavailable" in report


@pytest.fixture(autouse=True)
def _restore_device_module() -> Any:
    """Reload ``chamber.utils.device`` post-test so a stubbed torch doesn't leak.

    Mirrors the autouse fixture in :mod:`tests.unit.test_sapien_cuda_renderer_predicate`.
    """
    yield
    importlib.reload(device_module)
