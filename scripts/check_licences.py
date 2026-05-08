"""Gate dependency licences against the project allowlist.

Reads JSON from stdin (the output of ``pip-licenses --format=json``) and exits
non-zero if any dependency declares a licence that is not in
:data:`ALLOWED`. The allowlist matches plan/00-foundations.md §T0.12 and is
deliberately strict — any new licence has to land via an explicit ADR
amendment, not by quietly relaxing this list.

Usage:

    uv run pip-licenses --format=json | python scripts/check_licences.py
"""

from __future__ import annotations

import json
import sys

ALLOWED: frozenset[str] = frozenset(
    {
        "Apache-2.0",
        "Apache 2.0",
        "Apache Software License",
        "MIT",
        "MIT License",
        "BSD",
        "BSD License",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "ISC",
        "MPL-2.0",
        "Mozilla Public License 2.0 (MPL 2.0)",
        "PSF-2.0",
        "Python-2.0",
        "Python Software Foundation License",
        "CC0-1.0",
        "Public Domain",
        "Unlicense",
        "ZPL-2.1",
    }
)


def main(argv: list[str] | None = None) -> int:
    raw = sys.stdin.read()
    try:
        records = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"[licences] Could not parse JSON from stdin: {exc}", file=sys.stderr)
        return 2

    offenders: list[tuple[str, str, str]] = []
    for rec in records:
        name = rec.get("Name", "<unknown>")
        version = rec.get("Version", "<unknown>")
        licence = rec.get("License", "<unknown>")
        # pip-licenses sometimes emits multiple licences separated by semicolons.
        parts = [p.strip() for p in str(licence).split(";")]
        if not any(p in ALLOWED for p in parts):
            offenders.append((name, version, licence))

    if offenders:
        print("[licences] Disallowed licences detected:", file=sys.stderr)
        for name, version, licence in offenders:
            print(f"  - {name} {version}: {licence}", file=sys.stderr)
        return 1

    print(f"[licences] OK ({len(records)} dependencies, all in allowlist).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
