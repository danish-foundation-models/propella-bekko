"""Assemble the released distillation dataset (train / test-deepseek / test-*).

Text-free by design: rows keep the corpus coordinates (`source`, `subset`, `lang`,
`domain`, `id`) and a normalised `text_sha1`, so the text can be re-joined from the
public source corpora, but no document text is redistributed.

    uv run python scripts/make_dataset_release.py --out ../propella-bekko-dataset

Columns, identical in every split so one loader works for all of them:

    source, subset, lang, domain, id, text_sha1, n_chars, n_words,
    <17 propella-1-4b properties>, one_sentence_description,
    reference_annotator, ref_<17 properties>, ref_one_sentence_description

`reference_annotator` is null in `train` and names the independent annotator in the
`test-*` splits; the `ref_*` columns hold that annotator's labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from propella_bekko.schema import FIELD_NAMES

BASE = ["source", "subset", "lang", "domain", "id", "text_sha1", "n_chars", "n_words"]
TEACHER = [*FIELD_NAMES, "one_sentence_description"]
MULTI = {
    "content_type",
    "business_sector",
    "technical_content",
    "regional_relevance",
    "country_relevance",
}

# split name -> (docs parquet, reference label parquet or None, reference model id)
SPLITS = {
    "train": ("data/train-pool-xl.parquet", None, None),
    "test-deepseek": (
        "data/heldout-sample1000.parquet",
        "runs/reference/deepseek-v4.1-flash-1k.parquet",
        "deepseek-ai/DeepSeek-V4.1-Flash-1017",
    ),
    "test-glm-5.3-flash": (
        "data/heldout-sample1000.parquet",
        "runs/reference/glm-5.3-flash-pareto300.parquet",
        "zai-org/GLM-5.3-Flash",
    ),
}

TRAIN_SHARDS = 9


def column_schema() -> list[tuple[str, pa.DataType]]:
    out: list[tuple[str, pa.DataType]] = [
        ("source", pa.string()),
        ("subset", pa.string()),
        ("lang", pa.string()),
        ("domain", pa.string()),
        ("id", pa.string()),
        ("text_sha1", pa.string()),
        ("n_chars", pa.int64()),
        ("n_words", pa.int64()),
    ]
    for c in TEACHER:
        out.append((c, pa.list_(pa.string()) if c in MULTI else pa.string()))
    out.append(("reference_annotator", pa.string()))
    for c in TEACHER:
        out.append((f"ref_{c}", pa.list_(pa.string()) if c in MULTI else pa.string()))
    return out


def load_docs(path: str) -> pa.Table:
    return pq.read_table(path, columns=[*BASE, *TEACHER])


def build(docs: pa.Table, ref: pa.Table | None, model: str | None) -> pa.Table:
    n = docs.num_rows
    cols: dict[str, pa.ChunkedArray] = {c: docs.column(c) for c in [*BASE, *TEACHER]}
    cols["reference_annotator"] = pa.chunked_array([pa.array([model] * n, pa.string())])
    for c in TEACHER:
        if ref is None:
            cols[f"ref_{c}"] = pa.chunked_array([pa.nulls(n, cols[c].type)])
        else:
            cols[f"ref_{c}"] = ref.column(c)
    schema = pa.schema(column_schema())
    return pa.table([cols[f.name] for f in schema], schema=schema)


def join_reference(docs: pa.Table, ref_path: str, model: str, split: str) -> pa.Table:
    ref = pq.read_table(ref_path, columns=["source", "id", "_ok", *TEACHER])
    ref = ref.filter(ref.column("_ok"))
    keys = list(zip(ref.column("source").to_pylist(), ref.column("id").to_pylist(), strict=False))
    index = {k: i for i, k in enumerate(keys)}
    doc_keys = list(
        zip(docs.column("source").to_pylist(), docs.column("id").to_pylist(), strict=False)
    )
    keep = [i for i, k in enumerate(doc_keys) if k in index]
    print(
        f"  {split}: {len(keep)} docs joined, {len(doc_keys) - len(keep)} without a reference annotation"
    )
    docs = docs.take(pa.array(keep, pa.int64()))
    ref = ref.take(pa.array([index[doc_keys[i]] for i in keep], pa.int64()))
    return build(docs, ref, model)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="dataset repo checkout to write data/ into")
    p.add_argument("--train-shards", type=int, default=TRAIN_SHARDS)
    a = p.parse_args()

    data_dir = Path(a.out) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, int] = {}
    missing: dict[str, int] = {}

    for split, (docs_path, ref_path, model) in SPLITS.items():
        print(f"{split}: {docs_path}")
        docs = load_docs(docs_path)
        table = (
            join_reference(docs, ref_path, model, split) if ref_path else build(docs, None, None)
        )
        if split == "train":
            missing = {c: int(table.column(c).null_count) for c in TEACHER}
            step = -(-table.num_rows // a.train_shards)
            for i in range(a.train_shards):
                chunk = table.slice(i * step, step)
                if chunk.num_rows:
                    pq.write_table(chunk, data_dir / f"train-{i:05d}.parquet")
                    print(f"  wrote train-{i:05d}.parquet {chunk.num_rows} rows")
        else:
            pq.write_table(table, data_dir / f"{split}.parquet")
        summary[split] = table.num_rows

    (Path(a.out) / "stats.json").write_text(
        json.dumps({"rows": summary, "train_missing_labels": missing}, indent=2)
    )
    print(json.dumps({"rows": summary, "train_missing_labels": missing}, indent=2))


if __name__ == "__main__":
    main()
