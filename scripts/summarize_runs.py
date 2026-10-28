import json
import sys

for run in sys.argv[1:]:
    try:
        r = json.load(open(f"runs/{run}/results.json"))
    except FileNotFoundError:
        print(run, "not finished")
        continue
    h = r["heldout"]["summary"]
    print(
        f"{run:16s} acc {h['mean_single_accuracy']:.3f} jacc {h['mean_multi_jaccard']:.3f} "
        f"mF1 {h['mean_macro_f1']:.3f}  train {r['train_seconds'] / 60:.0f} min"
    )
