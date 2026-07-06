---
viewer: false
license: apache-2.0
pretty_name: CHAMBER-Bench v1.0 leaderboard bundles
tags:
  - robotics
  - reinforcement-learning
  - multi-agent
  - benchmark
---

# chamber-bench-leaderboard-bundles

The complete evidence behind every **CHAMBER-Bench v1.0** leaderboard
row, released with the
[fsafaei/concerto](https://github.com/fsafaei/concerto) repository. A
*result bundle* is a directory containing everything needed to check
a reported number: per-episode records, the exact producing command,
the preregistration it ran under, and SHA-256 integrity manifests
(bundle schema v3, ADR-028 in the repository).

## What is in this dataset

- `benchmark/cocarry-v1/`, `benchmark/handover-v1/` — the verbatim
  result-bundle directories listed in each task's
  `LEADERBOARD_BUNDLES.txt` (13 bundles), plus the campaign report
  and the committed checkpoint-selection artifacts (`selection/`).
  Each bundle contains `bundle.json`, `episodes_seed*.jsonl` (raw
  per-episode records), `partners.json`, `REPRO.txt` (the exact
  command), `prereg.yaml`, and `SHA256SUMS.txt`.
- `admission/` — the admission-report archive for each task-admission
  decision the rows depend on: `cocarry-2026-07-05` (ADMITTED),
  `handover_place-2026-07-05` (ADMITTED), and
  `stage1_pickplace_as-2026-07-05` (CONTROL — the protocol's
  falsification example). Each contains the human-readable report,
  the machine-readable `admission_report.json`, and one verifiable
  sub-bundle per check (A1 solvability, A2 single-robot
  infeasibility, A3 partner-relevance).
- `checkpoints/` — the content-addressed PyTorch checkpoints the
  learned rows (B-BLIND, B-AHT, B-JOINT) load via their
  `local://artifacts/...` URIs, with JSON sidecars.
- `manifest.json`, `SHA256SUMS.txt` — file list and digests; check a
  download with `sha256sum -c SHA256SUMS.txt`.

## How to verify

From a checkout of the repository, every bundle (including each
admission sub-bundle) re-verifies with one command that re-checks all
file hashes and recomputes the summary statistics (interquartile mean
and bootstrap confidence interval) from the raw records:

```bash
uv run chamber-eval verify <bundle-directory>
```

To re-run a row from scratch, place `checkpoints/` under
`artifacts/artifacts/` in the checkout and follow
`docs/how-to/reproduce-results.md` — CPU re-runs reproduce the
summary statistics exactly under the committed dependency lock.

## Reading the rows honestly

Two row labels are load-bearing: **REF-SCRIPT** is the *oracle
reference* (a scripted controller with privileged access — the
solvability ceiling, never a baseline), and **B-JOINT** is the
*non-AHT upper anchor* (a jointly-trained pair evaluated with its own
training partner, which violates the unseen-partner condition by
construction). Always carry these labels and the per-partner
breakdown when quoting numbers.

## Provenance

Generated in simulation by the repository's evaluation harness under
tag-locked preregistrations, July 2026 (admission runs under tags
`prereg-admission-cocarry-2026-07-05`,
`prereg-admission-handover-place-2026-07-05`,
`prereg-admission-stage1-pickplace-as-2026-07-05`). Each bundle
records its git commit, dependency-lock hash, platform fingerprint,
and seed schedule. No personal data of any kind.

## Limitations

Simulation only; two agents; the two admitted tasks and preregistered
baseline methods only; heterogeneity effects measured on co-carry are
null so far, so these rows support ad-hoc-teamwork competence claims,
not heterogeneity-robustness claims.

## Licence and citation

Apache-2.0, same as the repository. Cite via the repository's
`CITATION.cff` (DOI `10.5281/zenodo.20128468`). Maintainer contact
and deprecation policy: `MAINTENANCE.md` in the repository.
