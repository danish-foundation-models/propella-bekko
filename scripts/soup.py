"""Weight averaging (model soup) of student checkpoints that share an architecture.

Models trained from the same backbone init with the same head layout can be averaged
element-wise. Useful soups: all runs of one size class, or a greedy soup that adds a
checkpoint only if the held-out score improves.

    uv run python scripts/soup.py runs/soup-a25m --out runs/a25m-soup \
        runs/a25m-v1-pool/model runs/a25m-xl/model runs/a25m-langcap/model
"""

import json
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file


def soup(model_dirs: list[str], out_dir: str, weights: list[float] | None = None) -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sd = [load_file(str(Path(d) / "model.safetensors")) for d in model_dirs]
    weights = weights or [1.0] * len(sd)
    total = sum(weights)
    keys = set(sd[0])
    for other in sd[1:]:
        if set(other) != keys:
            raise RuntimeError("checkpoints have different parameter sets")
    merged = {}
    for k in keys:
        acc = sum(w * s[k].float() for w, s in zip(weights, sd, strict=True)) / total
        merged[k] = acc.to(sd[0][k].dtype).contiguous()
    save_file(merged, str(out / "model.safetensors"))
    for name in ("config.json",):
        shutil.copy2(Path(model_dirs[0]) / name, out / name)
    if (out / "tokenizer").exists():
        shutil.rmtree(out / "tokenizer")
    shutil.copytree(Path(model_dirs[0]) / "tokenizer", out / "tokenizer")
    cfg = json.loads((out / "config.json").read_text())
    (out / "soup.json").write_text(
        json.dumps(
            {"members": model_dirs, "weights": weights, "config": cfg.get("backbone")}, indent=2
        )
    )
    return str(out)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("models", nargs="+")
    a = ap.parse_args()
    print(soup(a.models, a.out))
