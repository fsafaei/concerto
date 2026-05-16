# Contributing to CONCERTO

Thank you for your interest in contributing.

## Before you start

- Read the [project principles](docs/explanation/principles.md) — P1 through P8 are non-negotiable.
- Every contribution must map to an ADR or an open issue. If you want
  to do something not covered by an ADR, open an `adr-question` issue first.
- Set your git identity for this repo:

  ```bash
  git config user.name  "Your Name"
  git config user.email "you@example.com"
  ```

## Development setup

```bash
git clone https://github.com/concerto-org/concerto.git
cd concerto
pip install uv
uv sync --group dev --group train
pre-commit install --install-hooks
```

`--group train` pulls the
[HARL fork](https://github.com/fsafaei/harl-fork) used by
`chamber.benchmarks.ego_ppo_trainer` and
`chamber.partners.frozen_harl`. It is *not* part of the runtime
dependencies because PyPI rejects wheel `METADATA` containing
`git+URL` direct references; see ADR-002 §Revision-history
(2026-05-16) and [#131](https://github.com/fsafaei/concerto/issues/131)
for the design rationale, and
[#132](https://github.com/fsafaei/concerto/issues/132) for the
Phase-1 follow-up that publishes the fork to PyPI.

## Workflow

1. Fork the repo and create a branch: `git checkout -b feat/<short-description>`.
2. Make your changes. Every new public symbol needs a docstring with an ADR
   reference (e.g. `# ADR-004 §6.2`). See P1.
3. Run `make verify` locally — all gates must be green before opening a PR.
4. Commit using [Conventional Commits](https://www.conventionalcommits.org/):
   `feat(envs): ...`, `fix(safety): ...`, `chore(deps): ...`, `docs(...)`.
5. Sign off: `git commit --signoff` (DCO). External contributors also need to
   sign the CLA (the CLA-assistant bot will prompt you on the first PR).
6. Open a PR against `main`. Use the [PR template](.github/PULL_REQUEST_TEMPLATE.md).

## Code style

- Formatter and linter: **ruff** (`make format` / `make lint`).
- Type checker: **pyright** strict mode (`make typecheck`).
- All style checks run automatically on `pre-commit`.

## Tests

```bash
make test               # full suite
uv run pytest tests/unit     # unit only
uv run pytest tests/property # property (Hypothesis)
uv run pytest -m smoke  # Stage-0 smoke (requires M1 complete)
```

Coverage gates: ≥80% overall; ≥95% on ADR-bearing modules.

The trainer / partner tests that touch the
[HARL fork](https://github.com/fsafaei/harl-fork) (e.g.
`tests/unit/test_partner_frozen_harl.py`,
`tests/unit/test_ego_ppo_trainer.py`,
`tests/unit/test_ego_ppo_trainer_rejects_non_frozen_partner.py`,
`tests/property/test_advantage_decomposition.py`,
`tests/integration/test_draft_zoo.py`) require the `train` dependency
group. `make install` installs it automatically; if you ran
`uv sync --group dev` only, those tests skip cleanly via
`pytest.importorskip` (each file declares a module-level skip naming
the install command). The
HARL lazy-import contract is pinned by
`tests/unit/test_harl_lazy_import.py`.

## Dependency changes

New runtime dependencies must:
1. Pass the licence allowlist in `scripts/check_licences.py` (Apache 2.0,
   MIT, BSD-*; no GPL/AGPL).
2. Have an ADR entry if load-bearing.
3. Pass `uv run pip-licenses` with no new red entries.

## Commit signing

Commits to `main` must be GPG- or SSH-signed. Set this up with:

```bash
git config commit.gpgSign true
```

## Bibliography hygiene

All public citations resolve through
[`docs/reference/refs.bib`](docs/reference/refs.bib). Maintainers and
contributors follow this checklist when touching any document that
cites a paper, standard, project, or model.

1. **Every acronym in public docs is resolved once.** A reader who
   searches `docs/reference/refs.bib` should find a full citation for
   every named method, benchmark, model, or standard appearing in the
   README, ADRs, explainer pages, or how-to guides.

2. **Foundational + recent.** Every major conceptual claim cites both
   a foundational work and a recent one. AHT, safety filtering,
   conformal calibration, and benchmark methodology each carry one
   canonical citation and one modern citation.

3. **Standards are separated from academic publications.**
   [`docs/reference/literature.md`](docs/reference/literature.md) is
   for peer-reviewed material;
   [`docs/reference/standards.md`](docs/reference/standards.md) is for
   normative documents; the international-evidence supplement is for
   industry signal.

4. **Exact edition / year / part number on every standards
   citation.** No `ISO 10218`; use `ISO 10218-2:2025`. No
   `Release 17`; use `3GPP TS 23.501 v17.x.y`.

5. **Internal-note references are mirrored publicly or marked
   internal-only.** Public ADRs never rely on a citation that exists
   only in a private `notes/` tree. If an ADR cites a reading note,
   that note's bibliographic record must also be in `refs.bib`.

6. **Benchmark claims disclose evaluation practice.** Seeds, bootstrap
   CI, rliable aggregate metrics, and pre-registration references are
   linked from
   [`docs/reference/evaluation.md`](docs/reference/evaluation.md).

7. **No new citation without a refs.bib entry first.** PRs that add
   or change a citation in a public doc must add or update the
   corresponding entry in `docs/reference/refs.bib` in the same PR.

8. **Optional adjacents are not preemptive.** Do not cite FMI, OMG
   DDS, NIST SP 800-82, or other adjacent specifications until the
   public scope makes a claim that depends on them.

## Questions?

Open an [issue](https://github.com/concerto-org/concerto/issues) or start
a [discussion](https://github.com/concerto-org/concerto/discussions).
