# SPDX-License-Identifier: Apache-2.0
"""Executable docs guard for the communication-stack snippet.

The docs claim a runnable ``encode -> decode`` round-trip against
``chamber.comm.fixed_format``. The snippet lived in the README
Quickstart until the CB-07 reviewer-path rewrite moved it to
``docs/explanation/architecture.md`` (the README quickstart is now the
``chamber-eval`` pair, guarded by ``scripts/check_readme_quickstart.py``
and the ``readme-quickstart`` CI job). If the snippet ever drifts from
the real public API, this test fails loudly under the ``smoke`` matrix.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOC = _REPO_ROOT / "docs" / "explanation" / "architecture.md"

# Match the first ```python ... ``` block that appears after the
# "## Try the communication stack" heading and before the next "## "
# section header.
_SNIPPET_SECTION_RE = re.compile(
    r"^##\s+Try the communication stack\b[^\n]*\n"  # the heading line
    r"(?P<section>.*?)"  # section body (non-greedy)
    r"(?=^##\s)",  # up to the next H2
    re.DOTALL | re.MULTILINE,
)
_PY_BLOCK_RE = re.compile(
    r"^```python\s*\n(?P<body>.*?)^```",
    re.DOTALL | re.MULTILINE,
)


def _extract_snippet_python(doc_text: str) -> str:
    section_match = _SNIPPET_SECTION_RE.search(doc_text)
    assert section_match is not None, "'Try the communication stack' section not found"
    block_match = _PY_BLOCK_RE.search(section_match.group("section"))
    assert block_match is not None, "no fenced ```python block under the snippet heading"
    return block_match.group("body")


@pytest.mark.smoke
def test_comm_snippet_executes(tmp_path: Path) -> None:
    """Exec the documented comm snippet and require non-empty stdout."""
    if shutil.which("uv") is None:
        pytest.skip("uv is not on PATH; skipping executable-docs check")

    body = _extract_snippet_python(_DOC.read_text(encoding="utf-8"))
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
        f"documented comm snippet exec failed (rc={result.returncode}).\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    assert result.stdout.strip(), "documented comm snippet produced no stdout"
