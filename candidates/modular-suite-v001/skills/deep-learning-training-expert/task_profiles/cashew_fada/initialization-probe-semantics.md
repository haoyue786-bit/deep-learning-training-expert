# Audited initialization diagnostic semantics

Use for pre-run discriminator tables, especially when interpreting step labels or comparing them with a prepared start. These facts apply to the audited producer only.

Producer: `initialization_probe` in the staged `train_expert_v4.py`, SHA256 `9bee8ef02671380c57d66ce49ec9789ffc75eedbf39b726df61454a11970388a`. Caller-supplied producer identity or a bounded check of the supplied source is needed before transferring this timing interpretation to another version. Without that identity, keep timing unknown.

The legacy table `D_lr_and_D_steps_probe` uses two cached batches, deep-copies fresh discriminators independently for each LR and discards all these states. Each loss/accuracy is computed before the optimizer step and appended after it. Thus its records labelled initial/after_2_steps/after_4_steps/after_8_steps describe states after 0/1/3/7 completed updates. They do not measure the state after the final eighth update. Preserve the original numbers and identify this interpretation; do not rewrite an earlier submitted packet or answer.

The additional eight warmup updates later in that function belong to a separate discarded gradient probe. They are not a common warm start for the LR branches. The table does not test discriminator updates per generator update: the predictor is fixed and no G update occurs. No held-out discriminator examples are evaluated by that legacy table.

Local accuracy marginalizes the eight output logits into the source/target groups at Stage4 locations; it is a domain metric, not eight-class accuracy or Cashew F1. Global accuracy is binary domain prediction per pooled Stage4 patch. Their observational units differ. Neither identifies target semantic alignment.

A different component-state measurement can explicitly report completed update counts and disjoint out-of-fit patch identities. Use its own recorded counts and population; do not apply the legacy minus-one conversion to it. Out-of-fit patches can still be spatially correlated. A prepared candidate includes its D parameters and AdamW history, predictor/model-optimizer identity, config and RNG. Having saved such a candidate is separate from choosing it or actually adopting it into main training.
