"""Independent reference labels from a large LLM via an OpenAI-compatible API.

Uses propella's detailed *annotator* prompt (the one propella used for its
non-propella baselines / Gemini ground truth) and its JSON schema, so the output
is directly comparable with propella-1 annotations.

    uv run python scripts/reference_labels.py --input runs/baselines/short200.parquet \
        --limit 50 --output runs/reference/deepseek-v4.1-flash-short50.parquet

The API key is read from $REFERENCE_API_KEY, falling back to the Hermes custom
provider key in ~/.hermes/.env (HERMES_CUSTOM_AI_CLOUD_SDU_DK_API_KEY). It is
never printed or written.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

ENV_KEY = "HERMES_CUSTOM_AI_CLOUD_SDU_DK_API_KEY"


def api_key() -> str:
    if os.environ.get("REFERENCE_API_KEY"):
        return os.environ["REFERENCE_API_KEY"]
    env = Path.home() / ".hermes" / ".env"
    for line in env.read_text().splitlines():
        if line.startswith(f"{ENV_KEY}="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit(f"no API key: set REFERENCE_API_KEY or {ENV_KEY} in {env}")


def load_propella(repo: str = "ellamind/propella-1-4b"):
    path = hf_hub_download(repo, "propella.py")
    hf_hub_download(repo, "property_descriptions.md")
    spec = importlib.util.spec_from_file_location("propella", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["propella"] = mod
    cwd = os.getcwd()
    os.chdir(os.path.dirname(path))
    try:
        spec.loader.exec_module(mod)
    finally:
        os.chdir(cwd)
    return mod


def parse_json(text: str) -> dict | None:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


async def annotate(rows, propella, a):
    key = api_key()
    schema = propella.get_annotation_response_schema(flatten=True, compact_whitespace=True)
    sem = asyncio.Semaphore(a.concurrency)
    out: list[dict] = [{} for _ in rows]
    usage = {"prompt_tokens": 0, "completion_tokens": 0}

    async with httpx.AsyncClient(
        base_url=a.base_url, headers={"Authorization": f"Bearer {key}"}, timeout=a.timeout
    ) as client:

        async def one(i, r):
            msgs = propella.create_annotator_messages(
                r["text"], max_content_chars=a.max_content_chars
            )
            body = {
                "model": a.model,
                "messages": msgs,
                "temperature": 0.0,
                "max_tokens": a.max_tokens,
            }
            if not a.no_schema:
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "AnnotationResponse", "schema": schema, "strict": True},
                }
            async with sem:
                for attempt in range(4):
                    try:
                        resp = await client.post("/chat/completions", json=body)
                        resp.raise_for_status()
                        d = resp.json()
                        ann = parse_json(d["choices"][0]["message"]["content"] or "")
                        for k in usage:
                            usage[k] += d.get("usage", {}).get(k, 0)
                        out[i] = {
                            "source": r["source"],
                            "id": r["id"],
                            "_ok": ann is not None,
                            **(ann or {}),
                        }
                        return
                    except (httpx.HTTPError, KeyError) as e:
                        if attempt == 3:
                            print(
                                f"failed {r['id']}: {type(e).__name__}: {str(e)[:200]}",
                                file=sys.stderr,
                            )
                            out[i] = {"source": r["source"], "id": r["id"], "_ok": False}
                        await asyncio.sleep(2**attempt)

        await asyncio.gather(*(one(i, r) for i, r in enumerate(rows)))
    return out, usage


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--base-url", default="https://ai.cloud.sdu.dk/v1")
    p.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash-1017")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--max-content-chars", type=int, default=50_000)
    p.add_argument("--no-schema", action="store_true", help="do not send response_format")
    p.add_argument("--timeout", type=float, default=600.0, help="per-request timeout (s)")
    a = p.parse_args()

    propella = load_propella()
    rows = pq.read_table(a.input, columns=["source", "id", "text"]).to_pylist()
    if a.limit:
        rows = rows[: a.limit]
    t0 = time.perf_counter()
    out, usage = asyncio.run(annotate(rows, propella, a))
    dt = time.perf_counter() - t0
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(out), a.output)
    stats = {
        "model": a.model,
        "docs": len(rows),
        "ok": sum(o.get("_ok", False) for o in out),
        "seconds": dt,
        **usage,
    }
    Path(a.output.replace(".parquet", ".json")).write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
