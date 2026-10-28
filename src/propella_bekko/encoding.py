"""Tokenisation / batching shared by training and inference."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from propella_bekko.data.sources import TRUNC_MARK
from propella_bekko.schema import Field, encode_labels


class Encoder:
    """Head+tail token truncation.

    Keeps the first `max_length - tail_tokens - 2` tokens and the last
    `tail_tokens` tokens, so the model sees both the beginning and whether the
    document ends cleanly (relevant for `content_integrity`).
    """

    def __init__(self, tokenizer, max_length: int = 1024, tail_tokens: int = 128):
        self.tok = tokenizer
        self.max_length = max_length
        self.tail_tokens = tail_tokens
        self.cls = (
            tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
        )
        self.sep = (
            tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
        )
        self.pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def ids(self, texts: list[str]) -> tuple[list[list[int]], list[bool]]:
        enc = self.tok(texts, add_special_tokens=False, truncation=False)["input_ids"]
        budget = self.max_length - 2
        out, trunc = [], []
        for t, ids in zip(texts, enc, strict=False):
            cut = len(ids) > budget
            if cut:
                tail = min(self.tail_tokens, budget // 2)
                ids = ids[: budget - tail] + ids[len(ids) - tail :]
            out.append([self.cls, *ids, self.sep])
            trunc.append(cut or TRUNC_MARK in t)
        return out, trunc

    def pad_batch(self, seqs: list[Any]) -> tuple[torch.Tensor, torch.Tensor]:
        lens = np.fromiter((len(s) for s in seqs), dtype=np.int32, count=len(seqs))
        L = int(lens.max()) if len(lens) else 0
        ids = np.full((len(seqs), L), self.pad, dtype=np.int32)
        for i, s in enumerate(seqs):
            ids[i, : lens[i]] = s if not isinstance(s, list) else np.asarray(s, dtype=np.int32)
        mask = (np.arange(L)[None, :] < lens[:, None]).astype(np.int32)
        return torch.from_numpy(ids.astype(np.int64)), torch.from_numpy(mask)

    def batch(self, rows: list[dict], fields: list[Field] | None = None) -> dict[str, Any]:
        texts = [r["text"] for r in rows]
        seqs, trunc = self.ids(texts)
        return self.collate(rows, seqs, trunc, fields)

    def collate(self, rows, seqs, trunc, fields=None) -> dict[str, Any]:
        ids, mask = self.pad_batch(seqs)
        b = {
            "input_ids": ids,
            "attention_mask": mask,
            "n_chars": torch.tensor([r.get("n_chars") or len(r["text"]) for r in rows]),
            "n_words": torch.tensor([r.get("n_words") or len(r["text"].split()) for r in rows]),
            "truncated": torch.tensor(trunc),
        }
        if fields is not None:
            enc = [encode_labels(r, fields) for r in rows]
            b["targets"] = {f.name: torch.tensor([e[f.name] for e in enc]) for f in fields}
        return b
