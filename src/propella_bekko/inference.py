"""Fast batched inference (annotation) with length-bucketed, token-budget batching."""

from __future__ import annotations

import time
from collections.abc import Iterator

import torch

from propella_bekko.encoding import Encoder
from propella_bekko.models import PropellaBekko


def token_budget_batches(
    lengths: list[int], max_tokens: int, max_batch: int = 512
) -> Iterator[list[int]]:
    """Group indices sorted by length so that batch_size * max_len <= max_tokens."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    batch: list[int] = []
    cur_max = 0
    for i in order:
        L = lengths[i]
        new_max = max(cur_max, L)
        if batch and (new_max * (len(batch) + 1) > max_tokens or len(batch) >= max_batch):
            yield batch
            batch, new_max = [], L
        batch.append(i)
        cur_max = new_max
    if batch:
        yield batch


class Annotator:
    def __init__(
        self,
        model: PropellaBekko,
        tokenizer,
        max_length: int = 1024,
        tail_tokens: int = 128,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        max_tokens: int = 65_536,
        threshold: float = 0.5,
        compile: bool = False,
    ):
        self.model = model.to(device).eval()
        if dtype != torch.float32:
            self.model = self.model.to(dtype)
        if compile:
            self.model = torch.compile(self.model, dynamic=True)
        self.enc = Encoder(tokenizer, max_length=max_length, tail_tokens=tail_tokens)
        self.device, self.max_tokens, self.threshold = device, max_tokens, threshold
        self._decode = model.decode

    @torch.inference_mode()
    def annotate(self, rows: list[dict]) -> list[dict]:
        seqs, trunc = self.enc.ids([r["text"] for r in rows])
        out: list[dict | None] = [None] * len(rows)
        for idx in token_budget_batches([len(s) for s in seqs], self.max_tokens):
            b = self.enc.collate(
                [rows[i] for i in idx], [seqs[i] for i in idx], [trunc[i] for i in idx]
            )
            b = {k: v.to(self.device, non_blocking=True) for k, v in b.items()}
            logits = self.model(**b)
            for i, ann in zip(idx, self._decode(logits, threshold=self.threshold), strict=False):
                out[i] = ann
        return [o for o in out if o is not None]

    def benchmark(self, rows: list[dict], warmup: int = 64) -> dict:
        self.annotate(rows[:warmup])
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        self.annotate(rows)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        return {
            "docs": len(rows),
            "seconds": dt,
            "docs_per_s": len(rows) / dt,
            "gpu_hours_per_1M_docs": 1e6 / (len(rows) / dt) / 3600,
        }
