.PHONY: install test lint typecheck format docs docs-build \
        verify-licences verify-no-ai-mentions sbom smoke verify

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
	uv run python scripts/check_licences.py

verify-no-ai-mentions:
	bash scripts/check_no_ai_mentions.sh

sbom:
	uv run cyclonedx-py environment -o sbom.spdx.json

smoke:
	uv run pytest -m smoke -x -v

verify: lint typecheck test docs-build verify-licences verify-no-ai-mentions
