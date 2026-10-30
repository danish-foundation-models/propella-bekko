"""Run a propella-1 model on a parquet of documents with vLLM (offline) and
measure throughput. Output columns match the propella annotation schema, so the
result can be scored with `propella-bekko`'s metrics against propella-1-4b labels.

Runs in a separate environment (vLLM pins its own torch), e.g.

    uv venv ~/envs/vllm && VIRTUAL_ENV=~/envs/vllm uv pip install vllm
    ~/envs/vllm/bin/python scripts/propella_baseline.py \
        --model ellamind/propella-1-0.6b --input data/heldout-small.parquet \
        --output runs/baselines/propella-1-0.6b.parquet

`propella.py` (prompts + JSON schema) is downloaded from the model repo.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


def load_propella(repo: str):
    path = hf_hub_download(repo, "propella.py")
    hf_hub_download(repo, "property_descriptions.md")
    spec = importlib.util.spec_from_file_location("propella", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["propella"] = mod
    cwd = os.getcwd()
    os.chdir(os.path.dirname(path))  # propella.py reads property_descriptions.md from cwd
    try:
        spec.loader.exec_module(mod)
    finally:
        os.chdir(cwd)
    return mod


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int)
    p.add_argument(
        "--sample", type=int, help="random (seeded) subset size; same subset for every model"
    )
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--max-content-chars", type=int, default=50_000)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--dtype", default="float16")  # Turing has no bf16
    p.add_argument(
        "--structured-backend",
        default="xgrammar",
        help="vLLM structured-output backend; the default avoids flashinfer's nvcc requirement",
    )
    a = p.parse_args()

    from vllm import LLM, SamplingParams

    # Newer vLLM picks a structured-output backend that may need nvcc to JIT; xgrammar does not.
    try:
        from vllm.config import StructuredOutputsConfig

        structured_cfg = StructuredOutputsConfig(backend=a.structured_backend)
    except ImportError:  # older vLLM: the default backend is already xgrammar
        structured_cfg = None

    propella = load_propella(a.model)
    rows = pq.read_table(a.input, columns=["source", "id", "text"]).to_pylist()
    if a.sample and a.sample < len(rows):
        import random

        rows = random.Random(0).sample(rows, a.sample)
    if a.limit:
        rows = rows[: a.limit]

    llm = LLM(
        model=a.model,
        dtype=a.dtype,
        max_model_len=a.max_model_len,
        gpu_memory_utilization=a.gpu_memory_utilization,
        enable_prefix_caching=True,
        **({"structured_outputs_config": structured_cfg} if structured_cfg else {}),
    )
    tok = llm.get_tokenizer()
    schema = propella.get_annotation_response_schema(flatten=True, compact_whitespace=True)
    # Schema-constrained decoding: the kwarg moved twice between vLLM releases.
    if hasattr(__import__("vllm.sampling_params", fromlist=["x"]), "StructuredOutputsParams"):
        from vllm.sampling_params import StructuredOutputsParams

        json_kwargs = {"structured_outputs": StructuredOutputsParams(json=schema)}
    elif hasattr(__import__("vllm.sampling_params", fromlist=["x"]), "GuidedDecodingParams"):
        from vllm.sampling_params import GuidedDecodingParams

        json_kwargs = {"guided_decoding": GuidedDecodingParams(json=schema)}
    else:
        json_kwargs = {"guided_json": schema}
    sp = SamplingParams(temperature=0.0, max_tokens=1024, **json_kwargs)
    prompts = []
    for r in rows:
        msgs = propella.create_messages(r["text"], max_content_chars=a.max_content_chars)
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        if len(ids) > a.max_model_len - 1024:  # keep head of the document
            keep = a.max_model_len - 1024 - 200
            text = tok.decode(tok(r["text"], add_special_tokens=False)["input_ids"][:keep])
            msgs = propella.create_messages(text, max_content_chars=a.max_content_chars)
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    llm.generate(prompts[:16], sp, use_tqdm=False)  # warmup
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp)
    dt = time.perf_counter() - t0

    recs, failed = [], 0
    for r, o in zip(rows, outs, strict=True):
        try:
            ann = json.loads(o.outputs[0].text)
        except json.JSONDecodeError:
            ann, failed = {}, failed + 1
        recs.append({"source": r["source"], "id": r["id"], **ann})
    pq.write_table(pa.Table.from_pylist(recs), a.output)
    stats = {
        "model": a.model,
        "docs": len(rows),
        "seconds": dt,
        "docs_per_s": len(rows) / dt,
        "failed_json": failed,
        "max_content_chars": a.max_content_chars,
    }
    print(json.dumps(stats, indent=2))
    with open(a.output.replace(".parquet", ".json"), "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
