# Decision log

Short records of decisions and why. Newest last.

## 001 — Single forward pass, one head per property
propella-1 generates a JSON object token by token. We instead predict every
property from one pooled document representation with a dedicated head:
softmax for single-select, sigmoid (BCE) for multi-select properties
(`content_type`, `business_sector`, `technical_content`, `regional_relevance`,
`country_relevance`). Multi-select decoding always emits at least one label
(propella never returns an empty list). `country_relevance` is open vocabulary:
the top-64 values in the training data + `other`.

## 002 — `one_sentence_description` is out of scope
Free text; not useful for filtering and incompatible with a classifier.

## 003 — Quality axis = agreement with propella-1-4b
All available labels come from propella-1-4b, so the reference for "quality" is
agreement with propella-1-4b on the held-out set. This compares fairly against
propella-1-0.6b/1.7b (which we run ourselves on the same documents); 4b scores
100% by construction. A stronger independent reference is a future issue.

## 004 — Start with the static model, then scale
Order: `bekko-static-embedding-v1` → `bekko-embedding-v1-a8m` → `-a25m`.
Small runs first for fast feedback, then scale data/model where it pays off.

## 005 — Length features
Students see a truncated document (head + tail tokens). Log char/word counts of
the *full* document and a truncation flag are concatenated to the pooled vector
so that `content_length` etc. remain learnable.

## 006 — Text sampling reads remote row groups
Several sources are tens of GB and the annotation shards total >100 GB, so data
builders read random parquet row groups over `hf://` (a few docs per group for
spread) and locate annotations by scanning only the `id` column, instead of
downloading whole shards.
