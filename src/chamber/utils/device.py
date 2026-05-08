# SPDX-License-Identifier: Apache-2.0
"""Hardware-availability utilities for CHAMBER.

ADR-001 §Risks: ManiSkill v3 requires Vulkan/GPU even in headless mode.
This module is the single authoritative source for device queries used by
the smoke script, pytest session header, and wrapper tests.  Each helper
is pure-query — no side effects, safe to call at import time.
"""

from __future__ import annotations

from typing import ClassVar


def sapien_gpu_available() -> bool:
    """Return True iff SAPIEN can initialise a Vulkan render system.

    ADR-001 §Risks: ManiSkill v3 requires a Vulkan-capable GPU.  This
    function is used to gate Tier-2 smoke tests and to report device
    status at session start.  Returns False on any exception (ImportError,
    RuntimeError from missing Vulkan, or anything else).
    """
    try:
        import mani_skill.envs  # noqa: F401
        from mani_skill.envs.sapien_env import BaseEnv

        _probe_robots: tuple[str, ...] = ("panda_wristcam",)

        class _VulkanProbe(BaseEnv):
            SUPPORTED_ROBOTS: ClassVar[list[tuple[str, ...]]] = [_probe_robots]  # type: ignore[assignment]

            def _load_scene(self, options: dict) -> None:
                pass

            def _initialize_episode(self, env_idx: object, options: dict) -> None:
                pass

        env = _VulkanProbe(  # type: ignore[arg-type, call-arg]
            robot_uids=_probe_robots,  # type: ignore[arg-type]
            num_envs=1,
            obs_mode="state",
            control_mode=None,
        )
        env.close()
        return True
    except Exception:
        return False


def torch_device() -> str:
    """Return the best available PyTorch device string.

    ADR-001 §Risks: used by M2+ training and evaluation code to select
    the compute backend automatically.  Priority: CUDA > MPS > CPU.
    Returns ``"cpu"`` if PyTorch is not installed.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def device_report() -> str:
    """Return a one-line human-readable device status for logs and CI output.

    ADR-001 §Risks: surfaces which hardware tier is active so that skipped
    Tier-2 tests are unambiguous in CI and local output.

    Example outputs::

        Hardware — SAPIEN/Vulkan: available   | PyTorch device: cuda
        Hardware — SAPIEN/Vulkan: unavailable | PyTorch device: cpu
    """
    sapien = "available  " if sapien_gpu_available() else "unavailable"
    pytorch = torch_device()
    return f"Hardware — SAPIEN/Vulkan: {sapien} | PyTorch device: {pytorch}"
