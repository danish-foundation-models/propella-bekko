.PHONY: install lint format lint-check type test check

install:  ## install the project + dev dependencies
	uv sync

format:  ## auto-format and fix lint issues
	uv run ruff format .
	uv run ruff check . --fix

lint: format type  ## format, lint and type-check

lint-check:  ## lint without modifying files (CI)
	uv run ruff format . --check
	uv run ruff check .

type:  ## static type checking with ty
	uv run ty check src

test:  ## run tests
	uv run pytest

check: lint-check type test  ## everything CI runs

help:
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'
