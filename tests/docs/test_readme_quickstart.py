# SPDX-License-Identifier: Apache-2.0
"""Executable docs guard for the README Quickstart block.

The README claims a runnable `encode -> decode` round-trip against
``chamber.comm.fixed_format``. If the snippet ever drifts from the
real public API, this test fails loudly under the ``smoke`` matrix.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_README = _REPO_ROOT / "README.md"

# Match the first ```python ... ``` block that appears after the
# "## Quickstart" heading and before the next "## " section header.
_QUICKSTART_PY_RE = re.compile(
    r"^##\s+Quickstart\b[^\n]*\n"  # the Quickstart heading line
    r"(?P<section>.*?)"  # section body (non-greedy)
    r"(?=^##\s)",  # up to the next H2
    re.DOTALL | re.MULTILINE,
)
_PY_BLOCK_RE = re.compile(
    r"^```python\s*\n(?P<body>.*?)^```",
    re.DOTALL | re.MULTILINE,
)


def _extract_quickstart_python(readme_text: str) -> str:
    section_match = _QUICKSTART_PY_RE.search(readme_text)
    assert section_match is not None, "README Quickstart section not found"
    block_match = _PY_BLOCK_RE.search(section_match.group("section"))
    assert block_match is not None, "no fenced ```python block under the Quickstart heading"
    return block_match.group("body")


@pytest.mark.smoke
def test_readme_quickstart_executes(tmp_path: Path) -> None:
    """Exec the README Quickstart python block and require non-empty stdout."""
    if shutil.which("uv") is None:
        pytest.skip("uv is not on PATH; skipping executable-docs check")

    body = _extract_quickstart_python(_README.read_text(encoding="utf-8"))
    script = tmp_path / "quickstart.py"
    script.write_text(body, encoding="utf-8")

    result = subprocess.run(  # noqa: S603
        ["uv", "run", "python", str(script)],  # noqa: S607
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        check=False,
    )
    assert result.returncode == 0, (
        f"README Quickstart exec failed (rc={result.returncode}).\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    assert result.stdout.strip(), "README Quickstart produced no stdout"
