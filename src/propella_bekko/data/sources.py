"""Build (text, propella-1-4b annotation) datasets.

Sources
-------
* **Dynaword** corpora (`danish-foundation-models/*-dynaword`): every subset with a
  `data/<subset>/metadata.parquet` is joined with `data/<subset>/data.parquet`.
  The subset's domain is read from the main table of the dataset README.
* **openeurollm/propella-annotations**: annotations without text. Texts are read
  from the source corpus (FineWeb-2 = web, FinePDFs = pdf) and joined on `id`.

Texts are always read remotely by random row groups (some subsets are tens of
GB). A few docs are taken from each row group so samples are spread out.

Output columns::

    source, subset, lang, domain, id, text, n_chars, n_words, text_sha1, <annotation fields>

`text` is truncated to `head_chars` + `tail_chars` characters (joined by
"\\n[...]\\n"); full-document length is preserved in `n_chars`/`n_words`.
"""

from __future__ import annotations

import hashlib
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download

from propella_bekko.schema import FIELD_NAMES

log = logging.getLogger(__name__)

ANNOTATION_REPO = "openeurollm/propella-annotations"
TRUNC_MARK = "\n[...]\n"
LABEL_COLUMNS = [*FIELD_NAMES, "one_sentence_description"]

# Dynaword corpora that currently ship propella-1-4b metadata.
DYNAWORDS = {
    "danish-foundation-models/danish-dynaword": "dan",
    "danish-foundation-models/norwegian-dynaword": "nor",
    "danish-foundation-models/icelandic-dynaword": "isl",
    "danish-foundation-models/faroese-dynaword": "fao",
    "danish-foundation-models/dutch-dynaword": "nld",
}

# corpus -> (text repo, domain, languages with propella annotations)
PROPELLA_CORPORA = {
    "fineweb-2": (
        "HuggingFaceFW/fineweb-2",
        "Web",
        ["deu_Latn", "fin_Latn", "fra_Latn", "ita_Latn", "spa_Latn", "swe_Latn"],
    ),
    "finepdfs": (
        "HuggingFaceFW/finepdfs",
        "PDF",
        [
            "ces_Latn", "dan_Latn", "deu_Latn", "eng_Latn", "fin_Latn", "fra_Latn", "hun_Latn",
            "ita_Latn", "nld_Latn", "nob_Latn", "pol_Latn", "por_Latn", "ron_Latn", "spa_Latn",
            "swe_Latn",
        ],
    ),
}  # fmt: skip


@dataclass
class TruncCfg:
    head_chars: int = 12_000
    tail_chars: int = 3_000


def truncate_text(text: str, cfg: TruncCfg) -> str:
    if len(text) <= cfg.head_chars + cfg.tail_chars + len(TRUNC_MARK):
        return text
    tail = text[-cfg.tail_chars :] if cfg.tail_chars > 0 else ""
    return text[: cfg.head_chars] + TRUNC_MARK + tail


def text_hash(text: str) -> str:
    return hashlib.sha1(" ".join(text.split()).encode()).hexdigest()


# --------------------------------------------------------------------------- #
# Remote sampling helpers
# --------------------------------------------------------------------------- #
_EMPTY = pa.schema([("id", pa.string()), ("text", pa.large_string())])


def sample_row_groups(
    files: list[str],
    n_docs: int,
    seed: int = 0,
    per_group: int = 32,
    exclude_ids: set[str] | None = None,
) -> pa.Table:
    """Sample ~n_docs (id, text) rows, at most `per_group` per random row group."""
    fs = HfFileSystem()
    rng = random.Random(seed)
    handles = {f: pq.ParquetFile(fs.open(f)) for f in files}
    groups = [(f, i) for f, h in handles.items() for i in range(h.metadata.num_row_groups)]
    rng.shuffle(groups)
    # few large row groups (small files) -> allow more docs per group
    per_group = max(per_group, -(-n_docs // max(1, len(groups))))
    excl = pa.array(list(exclude_ids), pa.string()) if exclude_ids else None
    out, n = [], 0
    for f, i in groups:
        t = handles[f].read_row_group(i, columns=["id", "text"]).cast(_EMPTY)
        if excl is not None:
            t = t.filter(pc.invert(pc.is_in(t.column("id"), excl)))
        k = min(per_group, t.num_rows, n_docs - n)
        if k <= 0:
            continue
        out.append(t.take(pa.array(sorted(rng.sample(range(t.num_rows), k)))))
        n += k
        if n >= n_docs:
            break
    return pa.concat_tables(out) if out else _EMPTY.empty_table()


def _finalize(
    texts: pa.Table,
    meta: pa.Table,
    source: str,
    subset: str,
    lang: str,
    domain: str,
    trunc: TruncCfg | None,
) -> pa.Table:
    """Join (id, text) with the annotation table on id and normalise columns."""
    trunc = trunc or TruncCfg()
    meta_cols = [c for c in LABEL_COLUMNS if c in meta.column_names]
    meta = meta.select(["id", *meta_cols])
    # pyarrow joins do not support list columns -> join by index
    pos = pc.index_in(texts.column("id"), value_set=meta.column("id").cast(pa.string()))
    keep = pc.is_valid(pos)
    texts = texts.filter(keep)
    meta = meta.take(pos.filter(keep))
    raw = texts.column("text").to_pylist()
    n = len(raw)
    cols = {
        "source": pa.array([source] * n, pa.string()),
        "subset": pa.array([subset] * n, pa.string()),
        "lang": pa.array([lang] * n, pa.string()),
        "domain": pa.array([domain] * n, pa.string()),
        "id": texts.column("id"),
        "text": pa.array([truncate_text(t, trunc) for t in raw], pa.large_string()),
        "n_chars": pa.array([len(t) for t in raw], pa.int64()),
        "n_words": pa.array([len(t.split()) for t in raw], pa.int64()),
        "text_sha1": pa.array([text_hash(t) for t in raw], pa.string()),
    }
    for c in meta_cols:
        col = meta.column(c)
        is_list = pa.types.is_list(col.type) or pa.types.is_large_list(col.type)
        cols[c] = col.cast(pa.list_(pa.string()) if is_list else pa.string())
    return pa.table(cols)


# --------------------------------------------------------------------------- #
# Dynaword
# --------------------------------------------------------------------------- #
def dynaword_subsets(repo: str) -> list[str]:
    files = HfApi().list_repo_files(repo, repo_type="dataset")
    subsets = {
        f.split("/")[1] for f in files if f.startswith("data/") and f.endswith("/metadata.parquet")
    }
    return sorted(s for s in subsets if f"data/{s}/data.parquet" in files)


def dynaword_domains(repo: str) -> dict[str, str]:
    """Parse `subset -> domain` from the main table of a Dynaword README."""
    readme = Path(hf_hub_download(repo, "README.md", repo_type="dataset")).read_text()
    m = re.search(r"<!-- START-MAIN TABLE -->(.*?)<!-- END-MAIN TABLE -->", readme, re.DOTALL)
    if not m:
        return {}
    out = {}
    for line in m.group(1).splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 3 and cells[0].startswith("[") and cells[0].endswith("]"):
            out[cells[0][1:-1]] = cells[2]
    return out


def build_dynaword(
    repo: str,
    n_per_subset: int,
    lang: str | None = None,
    subsets: list[str] | None = None,
    seed: int = 0,
    per_group: int = 32,
    exclude_ids: dict[tuple[str, str], set[str]] | None = None,
    trunc: TruncCfg | None = None,
) -> pa.Table:
    lang = lang or DYNAWORDS.get(repo, "unk")
    source = repo.split("/")[-1]
    domains = dynaword_domains(repo)
    tables = []
    for s in subsets or dynaword_subsets(repo):
        meta = pq.read_table(
            hf_hub_download(repo, f"data/{s}/metadata.parquet", repo_type="dataset")
        )
        texts = sample_row_groups(
            [f"datasets/{repo}/data/{s}/data.parquet"],
            n_per_subset,
            seed=seed,
            per_group=per_group,
            exclude_ids=(exclude_ids or {}).get((source, s)),
        )
        domain = domains.get(s, "Other")
        t = _finalize(texts, meta, source, s, lang, domain, trunc)
        log.info("%s/%s (%s): %d rows", source, s, domain, t.num_rows)
        tables.append(t)
    return pa.concat_tables(tables)


# --------------------------------------------------------------------------- #
# propella-annotations (+ source corpus texts)
# --------------------------------------------------------------------------- #
def annotation_files(corpus: str, lang: str) -> list[str]:
    fs = HfFileSystem()
    return sorted(
        fs.glob(f"datasets/{ANNOTATION_REPO}/data/propella-1-4b/{corpus}/{lang}/*.parquet")
    )


def fetch_annotations(files: list[str], wanted: pa.Array, threads: int = 16) -> pa.Table:
    """Find annotation rows for `wanted` ids without downloading whole shards.

    Phase 1 scans only the `id` column of every row group (in parallel);
    phase 2 reads full row groups that contain hits.
    """
    from concurrent.futures import ThreadPoolExecutor

    wanted = wanted.cast(pa.string())

    def n_groups(f):
        return pq.ParquetFile(HfFileSystem().open(f)).metadata.num_row_groups

    def scan(job):
        f, i = job
        ids = pq.ParquetFile(HfFileSystem().open(f)).read_row_group(i, columns=["id"]).column("id")
        return (f, i) if pc.any(pc.is_in(ids.cast(pa.string()), wanted)).as_py() else None

    def read(job):
        f, i = job
        t = pq.ParquetFile(HfFileSystem().open(f)).read_row_group(i)
        return t.filter(pc.is_in(t.column("id").cast(pa.string()), wanted))

    with ThreadPoolExecutor(threads) as ex:
        jobs = [
            (f, i) for f, n in zip(files, ex.map(n_groups, files), strict=False) for i in range(n)
        ]
        hits = [j for j in ex.map(scan, jobs) if j is not None]
        log.info(
            "annotation scan: %d/%d row groups contain sampled ids",
            len(hits),
            len(jobs),
        )
        tables = list(ex.map(read, hits))
    return pa.concat_tables(tables, promote_options="default")


def build_propella_annotations(
    corpus: str,
    lang: str,
    n_docs: int,
    seed: int = 0,
    per_group: int = 32,
    exclude_ids: dict[tuple[str, str], set[str]] | None = None,
    trunc: TruncCfg | None = None,
) -> pa.Table:
    repo, domain, _ = PROPELLA_CORPORA[corpus]
    source = f"propella-annotations/{corpus}"
    files = sorted(HfFileSystem().glob(f"datasets/{repo}/data/{lang}/train/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files for {repo} {lang}")
    # oversample: not every source doc has an annotation
    texts = sample_row_groups(
        files,
        int(n_docs * 1.3) + 10,
        seed=seed,
        per_group=per_group,
        exclude_ids=(exclude_ids or {}).get((source, lang)),
    )
    meta = fetch_annotations(annotation_files(corpus, lang), texts.column("id"))
    t = _finalize(texts, meta, source, lang, lang, domain, trunc)
    if t.num_rows > n_docs:
        t = t.take(pa.array(sorted(random.Random(seed).sample(range(t.num_rows), n_docs))))
    log.info(
        "%s/%s: %d rows (%d/%d texts annotated)",
        source, lang, t.num_rows, meta.num_rows, texts.num_rows,
    )  # fmt: skip
    return t


# --------------------------------------------------------------------------- #
# Held-out set / exclusion
# --------------------------------------------------------------------------- #
def heldout_ids(path: str | Path) -> tuple[dict[tuple[str, str], set[str]], set[str]]:
    """(source, subset) -> ids, plus text hashes, to exclude a held-out set from training."""
    t = pq.read_table(path, columns=["source", "subset", "id", "text_sha1"])
    by: dict[tuple[str, str], set[str]] = {}
    for s, sub, i in zip(
        t.column("source").to_pylist(),
        t.column("subset").to_pylist(),
        t.column("id").to_pylist(),
        strict=False,
    ):
        by.setdefault((s, sub), set()).add(i)
    return by, set(t.column("text_sha1").to_pylist())


def concat_dedup(tables: list[pa.Table], exclude_hashes: set[str] | None = None) -> pa.Table:
    t = pa.concat_tables(tables, promote_options="default")
    seen = set(exclude_hashes or ())
    keep = []
    for i, h in enumerate(t.column("text_sha1").to_pylist()):
        if h not in seen:
            seen.add(h)
            keep.append(i)
    if len(keep) < t.num_rows:
        log.info("dropped %d duplicate/held-out texts", t.num_rows - len(keep))
    return t.take(pa.array(keep))


def write(t: pa.Table, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(t, path, compression="zstd")
