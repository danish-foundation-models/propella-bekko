"""Command line interface: `propella-bekko {build-data,train,evaluate,annotate,benchmark}`."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path
from typing import Any


def _build_data(a):

    from propella_bekko.data import sources as D

    trunc = D.TruncCfg(a.head_chars, a.tail_chars)
    excl_ids, excl_hash = D.heldout_ids(a.exclude) if a.exclude else (None, None)
    import pyarrow.parquet as pq

    # every (source, subset) part is cached so an interrupted build resumes
    cache = Path(a.output).with_suffix("").parent / f".parts-{Path(a.output).stem}"
    cache.mkdir(parents=True, exist_ok=True)

    def cached(name: str, fn):
        part = cache / f"{name.replace('/', '__')}.parquet"
        if part.exists():
            return pq.read_table(part)
        t = fn()
        D.write(t, part)
        return t

    common: dict[str, Any] = dict(
        seed=a.seed, per_group=a.per_group, exclude_ids=excl_ids, trunc=trunc
    )
    labels_from = pq.read_table(a.labels_from) if a.labels_from else None
    import pyarrow.compute as _pc

    pc: Any = _pc
    tables = []
    dynawords = list(D.DYNAWORDS) if a.dynaword == ["all"] else a.dynaword
    for repo in dynawords:
        for sub in D.dynaword_subsets(repo):
            tables.append(
                cached(
                    f"{repo}/{sub}",
                    lambda repo=repo, sub=sub: D.build_dynaword(
                        repo, a.n_per_subset, subsets=[sub], **common
                    ),
                )
            )
    for spec in a.propella:
        corpus, lang = spec.split(":")
        langs = D.PROPELLA_CORPORA[corpus][2] if lang == "all" else [lang]
        for lg in langs:
            src = f"propella-annotations/{corpus}"
            lab = None
            if labels_from is not None:
                lab = labels_from.filter(
                    pc.and_(
                        pc.equal(labels_from.column("source"), src),
                        pc.equal(labels_from.column("subset"), lg),
                    )
                )
            tables.append(
                cached(
                    f"{corpus}/{lg}",
                    lambda corpus=corpus, lg=lg, lab=lab: D.build_propella_annotations(
                        corpus, lg, a.n_per_lang, labels=lab, **common
                    ),
                )
            )
    t = D.concat_dedup(tables, exclude_hashes=excl_hash)
    D.write(t, a.output)
    print(f"wrote {t.num_rows} rows -> {a.output}")
    for col in ("source", "lang", "domain"):
        vc = t.column(col).value_counts().to_pylist()
        print(
            f"{col}: "
            + ", ".join(
                f"{d['values']}={d['counts']}" for d in sorted(vc, key=lambda d: -d["counts"])
            )
        )


def _train(a):
    from propella_bekko.train import TrainConfig, train

    cfg = TrainConfig()
    if a.config:
        cfg = dataclasses.replace(cfg, **json.loads(Path(a.config).read_text()))
    over = {}
    for kv in a.set or []:
        k, v = kv.split("=", 1)
        try:
            over[k] = json.loads(v)
        except json.JSONDecodeError:
            over[k] = v
    cfg = dataclasses.replace(cfg, **over)
    r = train(cfg)
    print(json.dumps(r["val"]["summary"], indent=2))


def _load_rows(path, split, limit, sample=None):
    import pyarrow.parquet as pq

    filt = [("split", "=", split)] if split else None
    rows = pq.read_table(path, filters=filt).to_pylist()
    if sample and sample < len(rows):
        import random

        rows = random.Random(0).sample(rows, sample)  # same subset as scripts/propella_baseline.py
    return rows[:limit] if limit else rows


def _evaluate(a):
    from propella_bekko.inference import Annotator
    from propella_bekko.metrics import field_metrics, summarize
    from propella_bekko.models import PropellaBekko

    model, tok, cfg = PropellaBekko.load(a.model)
    rows = _load_rows(a.data, a.split, a.limit, a.sample)
    th = json.loads(Path(a.thresholds).read_text()) if a.thresholds else 0.5
    ann = Annotator(
        model, tok, cfg["max_length"], cfg["tail_tokens"], device=a.device, threshold=th
    )
    preds = ann.annotate(rows)
    res = {
        "summary": summarize(field_metrics(preds, rows, model.fields)),
        "fields": field_metrics(preds, rows, model.fields),
    }
    # breakdowns
    for col in ("source", "lang", "domain"):
        groups: dict[str, list[int]] = {}
        for i, r in enumerate(rows):
            groups.setdefault(r.get(col) or "?", []).append(i)
        res[f"by_{col}"] = {
            k: summarize(
                field_metrics([preds[i] for i in idx], [rows[i] for i in idx], model.fields)
            )
            | {"n": len(idx)}
            for k, idx in sorted(groups.items())
        }
    out = json.dumps(res, indent=2)
    if a.output:
        Path(a.output).write_text(out)
    print(json.dumps({k: v for k, v in res.items() if k != "fields"}, indent=2))


def _tune(a):
    from propella_bekko.tune import main as tune_main

    if a.output is None:
        a.output = str(Path(a.model).parent / "thresholds.json")
    tune_main(a)


def _score(a):
    """Score an annotation parquet (e.g. a propella-1 baseline) against held-out labels."""
    import pyarrow.parquet as pq

    from propella_bekko.metrics import field_metrics, summarize
    from propella_bekko.models import PropellaBekko

    gold = {(r["source"], r["id"]): r for r in pq.read_table(a.gold).to_pylist()}
    preds = [r for r in pq.read_table(a.pred).to_pylist() if (r["source"], r["id"]) in gold]
    golds = [gold[(r["source"], r["id"])] for r in preds]
    _, _, cfg = PropellaBekko.load(a.fields_from) if a.fields_from else (None, None, None)
    from propella_bekko.schema import fields_from_config
    from propella_bekko.train import build_fields

    fields = fields_from_config(cfg["fields"]) if cfg else build_fields(golds, 64)
    # unparseable generations count as wrong, not as missing
    for p in preds:
        for f in fields:
            p.setdefault(f.name, None)
            if p[f.name] is None:
                p[f.name] = "<invalid>" if f.kind == "single" else []
    res = field_metrics(preds, golds, fields)
    out = {"n": len(preds), "summary": summarize(res), "fields": res}
    if a.output:
        Path(a.output).write_text(json.dumps(out, indent=2))
    print(json.dumps({"n": out["n"], "summary": out["summary"]}, indent=2))


def _annotate(a):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from propella_bekko.inference import Annotator
    from propella_bekko.models import PropellaBekko

    model, tok, cfg = PropellaBekko.load(a.model)
    th = json.loads(Path(a.thresholds).read_text()) if a.thresholds else 0.5
    ann = Annotator(
        model,
        tok,
        cfg["max_length"],
        cfg["tail_tokens"],
        device=a.device,
        max_tokens=a.max_tokens,
        threshold=th,
    )
    pf = pq.ParquetFile(a.input)
    writer = None
    for batch in pf.iter_batches(batch_size=a.chunk_size, columns=[a.id_column, a.text_column]):
        rows = [
            {"id": i, "text": t}
            for i, t in zip(
                batch.column(a.id_column).to_pylist(),
                batch.column(a.text_column).to_pylist(),
                strict=False,
            )
        ]
        preds = ann.annotate(rows)
        out = pa.Table.from_pylist(
            [{"id": r["id"], **p} for r, p in zip(rows, preds, strict=False)]
        )
        if writer is None:
            writer = pq.ParquetWriter(a.output, out.schema, compression="zstd")
        writer.write_table(out)
    if writer:
        writer.close()
    print(f"wrote {a.output}")


def _benchmark(a):
    import torch

    from propella_bekko.inference import Annotator
    from propella_bekko.models import PropellaBekko, count_parameters

    model, tok, cfg = PropellaBekko.load(a.model)
    rows = _load_rows(a.data, a.split, a.limit, a.sample)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]
    ann = Annotator(
        model,
        tok,
        cfg["max_length"],
        cfg["tail_tokens"],
        device=a.device,
        dtype=dtype,
        max_tokens=a.max_tokens,
        compile=a.compile,
    )
    r = ann.benchmark(rows)
    r |= {
        "params_M": count_parameters(model) / 1e6,
        "max_length": cfg["max_length"],
        "dtype": a.dtype,
    }
    if torch.cuda.is_available():
        r["gpu"] = torch.cuda.get_device_name(0)
    print(json.dumps(r, indent=2))


def main(argv=None):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    for noisy in ("httpx", "httpx2", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    p = argparse.ArgumentParser("propella-bekko")
    sub = p.add_subparsers(required=True)

    b = sub.add_parser("build-data", help="build a (text, annotation) parquet (held-out or train)")
    b.add_argument(
        "--dynaword",
        nargs="*",
        default=[],
        help="dynaword repos, or 'all' for every annotated dynaword",
    )
    b.add_argument("--n-per-subset", type=int, default=100, help="docs per dynaword subset")
    b.add_argument(
        "--propella",
        nargs="*",
        default=[],
        help="CORPUS:LANG (LANG may be 'all'), e.g. fineweb-2:swe_Latn",
    )
    b.add_argument(
        "--n-per-lang",
        type=int,
        default=300,
        help="docs per propella-annotations language",
    )
    b.add_argument(
        "--per-group",
        type=int,
        default=10,
        help="max docs taken from one parquet row group",
    )
    b.add_argument("--exclude", help="held-out parquet whose ids/texts are excluded")
    b.add_argument(
        "--labels-from",
        help="earlier build with the same sampling params: reuse its propella-annotations "
        "labels (skips the annotation scan), e.g. to re-extract longer texts",
    )
    b.add_argument("--head-chars", type=int, default=12_000)
    b.add_argument("--tail-chars", type=int, default=3_000)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--output", required=True)
    b.set_defaults(fn=_build_data)

    t = sub.add_parser("train")
    t.add_argument("--config", help="JSON file with TrainConfig fields")
    t.add_argument("--set", nargs="*", help="override fields: key=value")
    t.set_defaults(fn=_train)

    for name, fn in [("evaluate", _evaluate), ("benchmark", _benchmark)]:
        e = sub.add_parser(name)
        e.add_argument("--model", required=True)
        e.add_argument("--data", required=True)
        e.add_argument("--split", default=None, help="optional value of a `split` column")
        e.add_argument("--limit", type=int)
        e.add_argument("--sample", type=int, help="seeded random subset (matches baselines)")
        e.add_argument("--thresholds", help="thresholds.json from tune-thresholds")
        e.add_argument("--device", default="cuda")
        if name == "evaluate":
            e.add_argument("--output")
        else:
            e.add_argument("--dtype", default="fp16")
            e.add_argument("--max-tokens", type=int, default=65_536)
            e.add_argument("--compile", action="store_true")
        e.set_defaults(fn=fn)

    tt = sub.add_parser(
        "tune-thresholds", help="tune multi-select decision thresholds on the val split"
    )
    tt.add_argument("--model", required=True)
    tt.add_argument("--data", required=True)
    tt.add_argument("--output", default=None)
    tt.add_argument("--limit", type=int)
    tt.add_argument("--val-frac", type=float, default=0.02)
    tt.set_defaults(fn=_tune)

    sc = sub.add_parser("score", help="score an annotation parquet against held-out labels")
    sc.add_argument("--pred", required=True)
    sc.add_argument("--gold", required=True)
    sc.add_argument("--fields-from", help="model dir whose label space (country vocab) to use")
    sc.add_argument("--output")
    sc.set_defaults(fn=_score)

    n = sub.add_parser("annotate", help="annotate a parquet file with id/text columns")
    n.add_argument("--model", required=True)
    n.add_argument("--input", required=True)
    n.add_argument("--output", required=True)
    n.add_argument("--id-column", default="id")
    n.add_argument("--text-column", default="text")
    n.add_argument("--chunk-size", type=int, default=20_000)
    n.add_argument("--max-tokens", type=int, default=65_536)
    n.add_argument("--device", default="cuda")
    n.add_argument("--thresholds", help="thresholds.json from tune-thresholds")
    n.set_defaults(fn=_annotate)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
