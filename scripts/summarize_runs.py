import json
import sys
from pathlib import Path

for run in sys.argv[1:]:
    try:
        r = json.loads(Path(f"runs/{run}/results.json").read_text())
    except FileNotFoundError:
        print(run, "not finished")
        continue
    h = r["heldout"]["summary"]
    t = r.get("heldout_tuned", {}).get("summary")
    extra = (
        f" | tuned acc {t['mean_single_accuracy']:.3f} mF1 {t['mean_macro_f1']:.3f}" if t else ""
    )
    print(
        f"{run:16s} acc {h['mean_single_accuracy']:.3f} jacc {h['mean_multi_jaccard']:.3f} "
        f"mF1 {h['mean_macro_f1']:.3f}{extra}  train {r['train_seconds'] / 60:.0f} min"
    )
