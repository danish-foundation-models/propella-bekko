"""Record one worked example per run: the full pbl_282 document and the model's answer."""

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
import torch
from propella_bekko.inference import Annotator
from propella_bekko.models import PropellaBekko

doc = pq.read_table("data/heldout-sample1000.parquet", columns=["id", "text"]).to_pylist()[0]

for run in sys.argv[1:]:
    path = Path("runs") / run / "model"
    model, tok, cfg = PropellaBekko.load(str(path))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ann = Annotator(
        model,
        tok,
        cfg["max_length"],
        cfg["tail_tokens"],
        device=device,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    out = ann.annotate([{"id": doc["id"], "text": doc["text"]}])[0]
    target = Path("runs") / run / "example_output.json"
    target.write_text(json.dumps({"id": doc["id"], "output": out}, indent=2))
    print(run, "->", target, "|", out["content_quality"], out["content_type"][:2])
