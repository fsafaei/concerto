# SPDX-License-Identifier: Apache-2.0
"""Parse-check every ``scripts/repro/*.sh`` via ``bash -n`` (plan/07 §5).

A pure-syntax guard so any future syntax error in a reproduction
script trips CI before someone wastes an afternoon discovering it on
a real Stage-1 GPU host. ``bash -n`` does **not** execute the
script; it only checks shell syntax.

Pinned at unit-test granularity (one parametrized case per script)
so a single bad script is named specifically in the failure report.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REPRO_DIR = _REPO_ROOT / "scripts" / "repro"

if shutil.which("bash") is None:  # pragma: no cover
    pytest.skip("bash binary not available on this host", allow_module_level=True)


def _repro_scripts() -> list[Path]:
    """Enumerate every ``scripts/repro/*.sh`` at collection time."""
    return sorted(_REPRO_DIR.glob("*.sh"))


@pytest.mark.parametrize("script_path", _repro_scripts(), ids=lambda p: p.name)
def test_repro_script_parses(script_path: Path) -> None:
    """``bash -n`` over the given script; pure-syntax check (plan/07 §5)."""
    bash = shutil.which("bash")
    assert bash is not None  # narrowed by module-level skip above.
    result = subprocess.run(  # noqa: S603 — bash binary resolved via shutil.which
        [bash, "-n", str(script_path)],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, (
        f"bash -n {script_path.name} returned {result.returncode}; stderr: {result.stderr.strip()}"
    )


def test_stage1_repro_scripts_exist() -> None:
    """Stage-1 AS and OM repro scripts are shipped (plan/07 §5)."""
    for axis in ("AS", "OM"):
        path = _REPRO_DIR / f"stage1_{axis.lower()}.sh"
        assert path.exists(), f"missing repro script {path}"
        assert path.stat().st_mode & 0o111, f"repro script {path} is not executable"
