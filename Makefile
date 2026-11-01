.PHONY: install lint format lint-check type test check data models figures checkout per-field sections publish annotate dataset help

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

data:  ## rebuild the held-out set, the 1k test sample and both training pools
	bash scripts/build_small.sh
	bash scripts/build_long.sh heldout
	uv run python scripts/make_pareto_sample.py
	bash scripts/build_pool_xl.sh
	bash scripts/build_long.sh pool

models:  ## retrain the three published students
	uv run propella-bekko train --config configs/static-long-ctx8192.json
	uv run propella-bekko train --config configs/a8m-xl.json
	uv run propella-bekko train --config configs/a25m-xl-3ep.json

annotate:  ## student predictions on the 1,000 held-out documents
	uv run python scripts/student_predict.py runs/static-long-ctx8192/model \
		data/heldout-sample1000.parquet runs/pareto/static-long-ctx8192.parquet
	uv run python scripts/student_predict.py runs/a8m-xl/model \
		data/heldout-sample1000.parquet runs/pareto/a8m-xl.parquet
	uv run python scripts/student_predict.py runs/a25m-xl-3ep/model \
		data/heldout-sample1000.parquet runs/pareto/a25m-xl-3ep.parquet

figures:  ## scores against both reference annotators, then the Pareto figures
	uv run python scripts/pareto_vs_llms.py
	uv run python scripts/pareto.py

checkout:  ## clone the published model and dataset repos this repo links (submodules)
	git submodule update --init --recursive

per-field:  ## per-property figures, written into each published model repo
	uv run python scripts/per_field.py --run a25m-xl-3ep --name propella-bekko-a25m \
		--out models/propella-bekko-a25m/per_property.svg
	uv run python scripts/per_field.py --run a8m-xl --name propella-bekko-a8m \
		--out models/propella-bekko-a8m/per_property.svg
	uv run python scripts/per_field.py --run static-long-ctx8192 --name propella-bekko-static \
		--out models/propella-bekko-static/per_property.svg

sections:  ## rewrite the marked regions of the cards and the README from the scores
	uv run python scripts/update_sections.py

publish:  ## push the model repos, then record their new commits here
	git -C models/propella-bekko-a25m push
	git -C models/propella-bekko-a8m push
	git -C models/propella-bekko-static push
	git add models dataset && git commit -m "Published model repos: current commits" || true

dataset:  ## rebuild and check the published dataset (a submodule)
	uv run python scripts/make_dataset_release.py --out dataset
	uv run python scripts/verify_dataset_release.py

help:
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'
