# Current A3 component roles depend on the loss path

Read when initialization or diagnosis relies on a component being active, inherited or untrained. These roles refer to the audited A3/soft-FADA implementation, not every model named Swin or A3. Identify the current producer before transferring them.

| Component | Classification | Model-side local/global FADA | Optimizer and measurement |
|---|---|---|---|
| root `norm` | Not traversed | Normalizes 3D Stage4 tokens; nonzero gradient dependency was observed | In model AdamW; excluded from the selected feature-gradient diagnostic vector |
| `fc1_DA`/`fc2_DA`/`fc3_DA` | Not traversed | Not traversed in these audited paths | Included in optimizer parameter groups but no gradients/history in the checked trajectory |
| A3 residual refiner | Used in segmentation | Its segmentation output supplies detached conditioning; traversal alone does not prove an adversarial gradient through A3 | New12 tensors; existing model-group multiplier5 is a contract, not an optimum |

A classification-only CPU hook record that calls the8 legacy tensors inactive is limited to that classification traversal. It is not a complete training-activity audit. The direct CPU graph check used a zero input and synthetic source labels; it establishes the observed dependency, not target quality or typical gradient magnitudes. Independently, the saved real E0/E1/E2 trajectory showed root-norm AdamW steps absent/63/126 and changed weights, while the6 legacy DA tensors remained unchanged with no optimizer history.

Do not equate checkpoint membership with source-pretraining history: the canonical bare checkpoint has no optimizer record proving each tensor's past updates. Preserve the demonstrated inherited prediction function. Do not reset the trained decoder merely because some auxiliary components are new or another checkpoint field is untrained.

The raw feature-gradient ratio describes the selected parameter vector, not all model-optimizer coordinates. Private adversarial parameters and classifier-only parameters can have different roles. This is a scope limitation, not a prescription to freeze normalization, change parameter groups or tune its learning rate. Use only advertised controls and a justified contrast.

Audited producer hashes: train_expert_v4.py `9bee8ef02671380c57d66ce49ec9789ffc75eedbf39b726df61454a11970388a`; architecture.py `48e19e25797c5c94a5028189c4f73e459adf7ab3ce195794bfa6e00d9894924a`; canonical GAN runtime `2362c66badb38851089703293a012a0f1e7f5247f9a9a52b37d2e2cedeb97436`. Other implementations require their actual graph/producer evidence.
