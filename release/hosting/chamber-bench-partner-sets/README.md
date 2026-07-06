---
license: apache-2.0
pretty_name: CHAMBER-Bench v1.0 partner sets
tags:
  - robotics
  - reinforcement-learning
  - multi-agent
  - benchmark
---

# chamber-bench-partner-sets

The versioned partner zoo of **CHAMBER-Bench v1.0** — the benchmark
for physically-coupled ad-hoc cooperation released with the
[fsafaei/concerto](https://github.com/fsafaei/concerto) repository. A
*partner* is the frozen, black-box teammate a benchmarked method is
evaluated against: the method controls one robot arm, the partner
controls the other, and the method never sees the partner's policy or
parameters.

## What is in this dataset

- `sets/*.json` — one machine-readable roster per partner-set version
  (`cocarry_partners@v1`, `cocarry_partners@v2`,
  `handover_place_partners@v1`, `stage1_pickplace_as_partners@v1`):
  set metadata, each member's construction box, split label, and
  SHA-256 identity hash. **Public** members carry their exact
  committed parameter values; **private** members carry `params:
  null` — their parameters are deliberately withheld.
- `cards/` — rendered per-member and per-set cards (construction,
  fingerprints, competence floors) as published in the project
  documentation.
- `fingerprints/*.json` — per-set behavioural fingerprint summaries.
- `checkpoints/` — the public learned members' PyTorch checkpoints
  (content-addressed, with JSON sidecars).
- `manifest.json`, `SHA256SUMS.txt` — file list and digests; check a
  download with `sha256sum -c SHA256SUMS.txt`.

## Why private members are hashes only

The benchmark keeps 30% of each partner set private so a submitted
method can be spot-checked against partners it cannot have overfitted.
Publishing the identity hashes (and the fingerprints measured from
probe episodes, released in `chamber-bench-reference-trajectories`)
makes that spot-check verifiable without revealing the parameters.
This is a benchmark-integrity measure; there is no personal or
confidential third-party data anywhere in this artifact.

## Provenance

All content is generated from the committed repository state by
`scripts/release/prepare_hosting.py` (deterministic; the producing
git commit is recorded in `manifest.json`). Rosters come from the
in-repo partner-set registry; fingerprints come from preregistered
probe runs archived as verifiable bundles. Collection is entirely
simulation and tooling output — no human subjects, no scraped data.

## Intended use

Instantiate the exact public partners behind a leaderboard row
(`uv run chamber-eval run --partner-set <set>@vN …`), evaluate new
methods against the public split, or audit partner identity. The
submission protocol is `docs/how-to/submit-leaderboard.md` in the
repository.

## Limitations

Scripted impedance controllers, scripted presenters, and frozen
jointly-trained policies only — no human models and no vision-based
partners in v1.0. Partner sets are versioned and never mutated; this
upload corresponds to the versions named in the roster files.

## Licence and citation

Apache-2.0, same as the repository. Cite via the repository's
`CITATION.cff` (DOI `10.5281/zenodo.20128469`). Maintainer contact
and deprecation policy: `MAINTENANCE.md` in the repository.
