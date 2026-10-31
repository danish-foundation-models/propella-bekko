# Configs

One config per run. `uv run propella-bekko train --config configs/<name>.json` writes
`runs/<output_dir>`.

Three of them are the published models:

| config | model |
|---|---|
| [`static-long-ctx8192.json`](static-long-ctx8192.json) | [propella-bekko-static](https://huggingface.co/danish-foundation-models/propella-bekko-static) |
| [`a8m-xl.json`](a8m-xl.json) | [propella-bekko-a8m](https://huggingface.co/danish-foundation-models/propella-bekko-a8m) |
| [`a25m-xl-3ep.json`](a25m-xl-3ep.json) | [propella-bekko-a25m](https://huggingface.co/danish-foundation-models/propella-bekko-a25m) |

The rest are the experiments behind [`experiments/`](../experiments/README.md), kept so that every
number in those notes can be re-run. The names are a trail: `v0` was the first run on a backbone,
`-pool` moved to the 403k pool, `-xl` to the 845k pool, and the suffixes mark what changed —
`pw` (position weight), `ls` (label smoothing), `domcap`/`langcap` (domain and language caps),
`clean` (the filtered pool), `swa`/`swa-ft` (checkpoint averaging), `refine` (label-conditional second
pass), `ord` (ordinal smoothing), `-c150` (a 150k-document subsample), `ctx<N>` (static-model context
length), `attn` (a learned token weighting instead of mean pooling).

`mmbert-*` and `sauerkraut-*` are the alternative backbones that were tried and did not pay off
(`experiments/011`).
