.PHONY: install lint format lint-check type test check data models figures cards annotate dataset help

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

cards:  ## regenerate the three model cards from docs/model_sheet.md
	uv run python scripts/example_output.py a25m-xl-3ep a8m-xl static-long-ctx8192
	uv run python scripts/model_card.py runs/a25m-xl-3ep propella-bekko-a25m --tier large \
		--docs-per-s 186 --mf1-deepseek 0.523 --mf1-glm 0.522 --n-train "844,912 documents"
	uv run python scripts/model_card.py runs/a8m-xl propella-bekko-a8m --tier small \
		--docs-per-s 348 --mf1-deepseek 0.489 --mf1-glm 0.479 --n-train "844,912 documents" \
		--train-minutes 92
	uv run python scripts/model_card.py runs/static-long-ctx8192 propella-bekko-static --tier static \
		--docs-per-s 686 --mf1-deepseek 0.479 --mf1-glm 0.468 --n-train "844,912 documents"

dataset:  ## rebuild and check the published dataset
	uv run python scripts/make_dataset_release.py --out ../propella-bekko-data
	uv run python scripts/verify_dataset_release.py

help:
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'
