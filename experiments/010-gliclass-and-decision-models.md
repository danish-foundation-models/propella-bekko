# 010 — what GLiClass and the recent "decision models" suggest for us

Read: GLiClass (arXiv 2508.07662, Knowledgator) and TypeSafe's System One models / Jev
(no paper; from their launch post and third-party analyses).

## Jev (System One)

Typed decisions instead of text: `Choice` (softmax over a declared option set), `Score`
(continuous value on an ordered rubric, *with the underlying distribution*), `Noul` (calibrated
truth probability). One parallel pass, no decoding loop; training uses RLCD, which penalises
confident mistakes more than uncertain ones.

Transferable to us, in order of cost:

1. **Decode the distribution, not the argmax** — adopted. Our ordinal properties are `Score`
   decisions, and we were taking `argmax`. Decoding to the *expected* level
   (`round(Σ p_i · rank_i)`) lifts mean QWK from **0.8116 to 0.8243** (+0.013) and the paper
   score from **0.811 to 0.819** on the 1k test set, with no training at all
   (`scripts/soft_ordinal.py`, now the default `ordinal_decode="expected"`). Biggest gains:
   content_integrity +0.049, commercial_bias +0.023, time_sensitivity +0.024.
2. **Calibrated probabilities as an output contract** — adopted for reporting: every prediction
   already carries a per-field distribution, so we can publish confidence and (later) an ECE /
   Brier column in the card. Opting not to train an RLCD-style objective: no published loss,
   and our labels are a single teacher's, so calibration would be calibrated *to the teacher*.
3. **Atomic parallel questions over one shared state** — already our architecture (one document
   encoder, 17 independent heads, no cross-field conditioning). Jev's design argues for keeping
   it that way rather than modelling field dependencies.

## GLiClass

`«LABEL»` token + label text concatenated with the document in one encoder pass, then pooling
and a dot-product or small-MLP scorer between the pooled text and pooled label embeddings;
plus squeeze-excitation layer re-weighting, a token-level contrastive loss, and PPO for
data-sparse multi-label training.

1. **Label names as the classifier rows** — adopted in the cheap form. Rather than paying to
   put ~600 label tokens through the encoder for every document (which would cost most of our
   throughput), `label_init` initialises each output row with the pooled embedding of that
   label's own name ("advertising marketing", "iceland"). Rare classes — 520 country names,
   the tail of the sector ontology — start from their meaning instead of from noise, and are
   fine-tuned normally afterwards. Queued as `a25m-v3`.
2. **Layer-wise re-weighting (squeeze-excitation)** — cheap in parameters (one scalar per layer)
   but needs per-layer hidden states in the encoder wrapper; deferred until the v2/v3 results
   show whether head/feature changes are still moving the score.
3. **Token-level contrastive loss, PPO** — not adopted. Both are aimed at few-shot / data-sparse
   and human-feedback settings; we have ~845k labelled documents, so extra loss terms would
   buy little and complicate the recipe.

## Speed context

GLiClass's published throughput is 97 examples/s for their 32.7M `edge` model and 25/s for the
439M `large` on their benchmark. Our 25M-active model runs at ~193 docs/s on 20k-character
documents, so we are already ahead of that trade-off; the interesting borrowings are the ones
that improve quality at zero throughput cost (distribution decoding, label semantics in the
output rows).
