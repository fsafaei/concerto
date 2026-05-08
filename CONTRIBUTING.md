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
uv sync --group dev
pre-commit install --install-hooks
```

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

## Questions?

Open an [issue](https://github.com/concerto-org/concerto/issues) or start
a [discussion](https://github.com/concerto-org/concerto/discussions).
