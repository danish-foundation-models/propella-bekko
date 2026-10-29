"""Upload a trained student (and its README) to the Hub.

    uv run python scripts/publish.py runs/a25m-xl danish-foundation-models/propella-bekko-a25m

The model directory holds `model/` (checkpoint), `thresholds.json` and `results.json`;
only the checkpoint is uploaded. A model card is generated from `experiments/` numbers
passed with --card-data (see scripts/model_card.md).
"""

import json
import sys
from pathlib import Path

from huggingface_hub import HfApi

run_dir, repo_id = sys.argv[1:3]
run = Path(run_dir)
model_dir = run / "model"
api = HfApi()

api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
api.upload_folder(
    folder_path=str(model_dir),
    repo_id=repo_id,
    repo_type="model",
    commit_message=f"Upload {run.name}",
)
print(f"uploaded {model_dir} -> https://huggingface.co/{repo_id}")

th = run / "thresholds.json"
if th.exists():
    api.upload_file(
        path_or_fileobj=str(th),
        path_in_repo="thresholds.json",
        repo_id=repo_id,
        repo_type="model",
    )
res = run / "results.json"
if res.exists():
    d = json.loads(res.read_text())
    api.upload_file(
        path_or_fileobj=str(res),
        path_in_repo="results.json",
        repo_id=repo_id,
        repo_type="model",
    )
    print("held-out:", d.get("heldout", {}).get("summary"))
    print("held-out (tuned):", d.get("heldout_tuned", {}).get("summary"))
