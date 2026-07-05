.PHONY: install test lint typecheck format docs docs-build \
        verify-licences verify-no-ai-mentions verify-coverage-floors \
        verify-changelog-completeness verify-readme-tables \
        sbom smoke smoke-eval verify release-preflight \
        empirical-guarantee zoo-seed-gpu zoo-seed-pull zoo-seed-verify \
        stage1-as stage1-om render-task-docs

install:
	# HARL ships as the `harl-aht` PyPI distribution (ADR-002
	# §Revision-history 2026-05-19, supersedes the 2026-05-16
	# `[dependency-groups].train` workaround per #132). It resolves
	# automatically as a runtime dependency, so no `--group train`
	# step is required.
	uv sync --all-extras --group dev

test:
	uv run pytest

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

format:
	uv run ruff check --fix src tests
	uv run ruff format src tests

typecheck:
	uv run pyright

docs:
	uv run mkdocs serve

docs-build:
	uv run mkdocs build --strict

verify-licences:
	uv run pip-licenses --format=json --with-license-file --with-notice-file \
	    --output-file=licences.json
	uv run python scripts/check_licences.py < licences.json

verify-no-ai-mentions:
	bash scripts/check_no_ai_mentions.sh

# Guard against the release-please CHANGELOG-skip bug (#126). Re-runs a
# minimal Conventional-Commits filter over `git log <prev-tag>..HEAD` and
# asserts every release-worthy commit's short SHA appears in the top
# CHANGELOG.md section. Silent on non-release branches (top section already
# matches the latest tag → nothing to verify). Load-bearing on the
# release-please PR.
verify-changelog-completeness:
	uv run python scripts/check_changelog_completeness.py

# Per-package line-coverage floors (plan/02 §6 #2; plan/03 §6 #8;
# plan/04 §6 #5; plan/05 §6 #8; plan/06 §6 #6). Reads coverage.xml
# emitted by `make test` — runs after it in the verify chain so the
# report is fresh. Project-wide aggregate floor (80%) is still
# enforced by pyproject.toml's `[tool.coverage.report] fail_under`.
verify-coverage-floors:
	uv run python scripts/check_coverage_floors.py

# Drift gate for the generated benchmark-suite surfaces (ADR-027
# §Versioning / §Consequences): the README task table and the
# docs/reference/tasks/ cards must match a fresh render from the
# chamber.tasks registry. Regenerate with `make render-task-docs`.
verify-readme-tables:
	uv run python scripts/render_task_table.py --check
	uv run python scripts/render_task_cards.py --check

# Regenerate the README task table + docs/reference/tasks/ cards from
# the chamber.tasks registry (the write mode of verify-readme-tables).
render-task-docs:
	uv run python scripts/render_task_table.py
	uv run python scripts/render_task_cards.py

sbom:
	uv run cyclonedx-py environment -o sbom.spdx.json

# Pre-upload gate for a PyPI / TestPyPI release; mirrors the build job in
# .github/workflows/release.yml. Builds fresh artefacts, validates the
# long-description render (twine) and the release metadata contract, and
# checks every runtime dep resolves from production PyPI (a TestPyPI-only
# dep would break `pip install concerto-multirobot` for end users).
# twine runs ephemerally via uvx — it stays out of the project
# dependency tree (its transitive docutils trips the ADR-012 licence
# gate, and a release tool has no business in the wheel's environment).
# twine >=6.1 is the floor that understands Metadata-Version 2.4.
release-preflight:
	rm -rf dist
	uv build
	uvx --from 'twine>=6.1' twine check --strict dist/*
	uv run python scripts/check_dist_metadata.py --check-resolvable

smoke:
	uv run pytest -m smoke -x -v

# ADR-028 §Validation criteria 1: run → verify → tamper → verify-fails,
# end to end through the chamber-eval CLI on the Tier-0 CPU task.
# On a dirty local tree: SMOKE_EVAL_ALLOW_DIRTY=1 make smoke-eval
smoke-eval:
	bash scripts/smoke_eval.sh

empirical-guarantee:
	uv run pytest tests/integration/test_empirical_guarantee.py -m slow --no-cov -v

zoo-seed-gpu:
	bash scripts/repro/zoo_seed.sh

# Fetch the published zoo-seed artefact + sidecar onto any host so the
# verify-only path (zoo-seed-verify) can re-check integrity without
# re-running the 2 h GPU training (plan/05 §6 #5).
#
# TODO(maintainer): replace ZOO_SEED_BASE_URL with the canonical hosting
# location once the artefact is uploaded (Zenodo is the natural home;
# hosting is a separate v1.0 work item). Until then this
# target prints a friendly error so a copy-paste invocation does not
# silently leak a placeholder URL into someone's shell history.
ZOO_SEED_BASE_URL ?=
ZOO_SEED_NAME := happo_seed7_step50k.pt
ZOO_SEED_DEST_DIR := artifacts/artifacts
zoo-seed-pull:
	@if [ -z "$(ZOO_SEED_BASE_URL)" ]; then \
		echo "zoo-seed-pull: ZOO_SEED_BASE_URL is unset."; \
		echo "  Set it on the maintainer's hosting URL, e.g.:"; \
		echo "  make zoo-seed-pull ZOO_SEED_BASE_URL=https://zenodo.org/record/<NNN>/files"; \
		echo "  (The target fetches both the .pt and the .pt.json sidecar.)"; \
		exit 2; \
	fi
	mkdir -p $(ZOO_SEED_DEST_DIR)
	curl -fsSL "$(ZOO_SEED_BASE_URL)/$(ZOO_SEED_NAME)" -o "$(ZOO_SEED_DEST_DIR)/$(ZOO_SEED_NAME)"
	curl -fsSL "$(ZOO_SEED_BASE_URL)/$(ZOO_SEED_NAME).json" -o "$(ZOO_SEED_DEST_DIR)/$(ZOO_SEED_NAME).json"
	@echo "Fetched $(ZOO_SEED_NAME) + sidecar under $(ZOO_SEED_DEST_DIR)."
	@echo "Run \`make zoo-seed-verify\` to re-check the SHA-256 manifest."

zoo-seed-verify:
	uv run pytest tests/reproduction/test_zoo_seed_artifact.py -v --no-cov

# Stage-1 AS / OM reproduction shells (ADR-007 §Implementation staging
# Stage 1; plan/07 §5). Each target pre-flights the on-disk prereg
# against its committed git tag (ADR-007 §Discipline), then runs the
# chamber-side adapter + the ADR-008 evaluation pipeline. The
# resulting SpikeRun + leaderboard artefacts land under
# spikes/results/stage1-<axis>-<date>/.
stage1-as:
	bash scripts/repro/stage1_as.sh

stage1-om:
	bash scripts/repro/stage1_om.sh

verify: lint typecheck test verify-coverage-floors docs-build verify-licences verify-no-ai-mentions verify-changelog-completeness verify-readme-tables
