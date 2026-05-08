"""Gate dependency licences against the project allowlist.

Reads JSON from stdin (the output of ``pip-licenses --format=json``) and exits
non-zero if any dependency declares a licence that is not in
:data:`ALLOWED`. The allowlist matches plan/00-foundations.md §T0.12 and is
deliberately strict — any new licence has to land via an explicit ADR
amendment, not by quietly relaxing this list.

Usage:

    uv run pip-licenses --format=json | python scripts/check_licences.py

Project policy (ADR-012): Apache 2.0 only; no GPL/AGPL deps.
LGPL is permitted for linking (not a copyleft trigger on our source).
"""

from __future__ import annotations

import json
import re
import sys

# Canonical SPDX identifiers and the common human-readable variants emitted by
# pip-licenses for each family.  Extend only via an ADR-012 amendment.
ALLOWED: frozenset[str] = frozenset(
    {
        # Apache
        "Apache-2.0",
        "Apache 2.0",
        "Apache Software License",
        "Apache License, Version 2.0",
        "Apache License 2.0",
        # MIT
        "MIT",
        "MIT License",
        "MIT-CMU",  # PIL/Pillow variant
        # BSD
        "BSD",
        "BSD License",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "3-Clause BSD License",
        "BSD 3-Clause",
        "BSD 2-Clause",
        "Modified BSD License",
        "New BSD License",
        "Simplified BSD License",
        # ISC — functionally equivalent to MIT; OSI-approved
        "ISC",
        "ISC License",
        "ISC License (ISCL)",
        # MPL-2.0
        "MPL-2.0",
        "Mozilla Public License 2.0 (MPL 2.0)",
        # PSF / Python
        "PSF-2.0",
        "Python-2.0",
        "Python Software Foundation License",
        "PSFL",
        # LGPL — permitted for linking (ADR-012: bans GPL/AGPL only, not LGPL)
        "LGPL-2.0",
        "LGPL-2.1",
        "LGPL-3.0",
        "GNU Lesser General Public License v2 (LGPLv2)",
        "GNU Lesser General Public License v2 or later (LGPLv2+)",
        "GNU Lesser General Public License v3 (LGPLv3)",
        "GNU Lesser General Public License v3 or later (LGPLv3+)",
        "GNU Library or Lesser General Public License (LGPL)",
        # Public domain / permissive
        "CC0-1.0",
        "Public Domain",
        "Unlicense",
        "ZPL-2.1",
        # Other OSI-approved permissive
        "Artistic-2.0",
        "EUPLv1.2",
    }
)

# Packages whose pip-licenses metadata reports UNKNOWN or a non-standard string
# but whose actual licence has been manually verified as allowlist-compatible.
# Format: "Name==version"  (exact match to pip-licenses Name field).
KNOWN_CLEAN: frozenset[str] = frozenset(
    {
        "mani_skill==3.0.1",  # Apache-2.0 (verified from source repo)
        "matplotlib-inline==0.2.1",  # BSD-3-Clause (verified from source repo)
    }
)

# NVIDIA CUDA runtime libraries installed as transitive dependencies of PyTorch
# (ADR-002). These are GPU compute libs distributed under the NVIDIA EULA;
# they are not linked into CONCERTO source code and do not affect the
# project's open-source licence obligations. Matched by name only (no version
# pin) so this list stays valid across CUDA minor-version bumps.
PROPRIETARY_RUNTIME_EXEMPTIONS: frozenset[str] = frozenset(
    {
        "nvidia-cublas-cu12",
        "nvidia-cuda-cupti-cu12",
        "nvidia-cuda-nvrtc-cu12",
        "nvidia-cuda-runtime-cu12",
        "nvidia-cudnn-cu12",
        "nvidia-cufft-cu12",
        "nvidia-curand-cu12",
        "nvidia-cusolver-cu12",
        "nvidia-cusparse-cu12",
        "nvidia-nccl-cu12",
        "nvidia-nvjitlink-cu12",
        "nvidia-nvtx-cu12",
    }
)

# Separator for compound SPDX licence expressions like "Apache-2.0 OR BSD-2-Clause"
# or "MPL-2.0 AND MIT".  Uppercase-only to avoid matching "or later" inside
# human-readable LGPL names such as "GPL v2 or later".
_COMPOUND_RE = re.compile(r"\s+(?:OR|AND)\s+")


def _is_allowed(licence: str) -> bool:
    """Return True if every component of a compound licence expression is allowed."""
    parts = [p.strip() for p in _COMPOUND_RE.split(licence)]
    parts = [item for p in parts for item in p.split(";")]
    return all(p.strip() in ALLOWED for p in parts if p.strip())


def main() -> int:
    """Read pip-licenses JSON from stdin and exit non-zero on disallowed licences."""
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

        if f"{name}=={version}" in KNOWN_CLEAN:
            continue
        if name in PROPRIETARY_RUNTIME_EXEMPTIONS:
            continue

        if licence == "UNKNOWN":
            offenders.append((name, version, "UNKNOWN (unverified)"))
            continue

        if not _is_allowed(licence):
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
