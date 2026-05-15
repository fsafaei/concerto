.PHONY: install test lint typecheck format docs docs-build \
        verify-licences verify-no-ai-mentions sbom smoke verify \
        empirical-guarantee zoo-seed-gpu zoo-seed-pull zoo-seed-verify

install:
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

sbom:
	uv run cyclonedx-py environment -o sbom.spdx.json

smoke:
	uv run pytest -m smoke -x -v

empirical-guarantee:
	uv run pytest tests/integration/test_empirical_guarantee.py -m slow --no-cov -v

zoo-seed-gpu:
	bash scripts/repro/zoo_seed.sh

# Fetch the published zoo-seed artefact + sidecar onto any host so the
# verify-only path (zoo-seed-verify) can re-check integrity without
# re-running the 2 h GPU training (plan/05 §6 #5).
#
# TODO(maintainer): replace ZOO_SEED_BASE_URL with the canonical hosting
# location once the artefact is uploaded (Zenodo is the natural home —
# see the existing `zenodo_announcement/` folder). Until then this
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

verify: lint typecheck test docs-build verify-licences verify-no-ai-mentions
