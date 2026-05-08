# Project principles

These eight principles govern all design and implementation decisions.
When a milestone description conflicts with a principle, the principle wins.

**P1 — ADR-traceability is mandatory.**
Every public function, class, and config key in an ADR-bearing module
carries an ADR reference in its docstring. CI fails on unreferenced public
symbols.

**P2 — Wrapper layer, never monkey-patch.**
ManiSkill v3 is a pinned external dependency. We never modify its internals
and never import `_private` symbols. If a genuine upstream change is needed,
we upstream a PR.

**P3 — Verification before completion.**
No task is marked done until its machine-checkable acceptance criterion
passes. `make verify` is the canonical check.

**P4 — Pre-register, then run.**
For all ADR-007 spikes, the comparison protocol is committed and tagged
before any training run launches. Post-hoc threshold adjustment is a
project anti-pattern.

**P5 — Open-source-grade from day one.**
Apache 2.0, SPDX headers, DCO/CLA, dependabot, OSSF Scorecard, signed
commits, Conventional Commits, Diátaxis docs, and OSI-clean dependencies
from the first commit.

**P6 — Reproducibility ≥ convenience.**
Every training run and spike is reproducible from `uv.lock` + config hash
+ seed. `scripts/repro/<artifact>.sh` is the canonical reproduction entry
point.

**P7 — Modular, with named extension points.**
A small core (env wrappers, safety filter, partner interface, training loop,
evaluation harness) plus plug-in registries for everything that varies per
experiment. New behaviour adds a registry entry — never a switch in the core.

**P8 — Documentation parity with code.**
Code is not done until its mkdocs page exists. New public symbols land in
the same PR as their docstring and mkdocstrings page.
