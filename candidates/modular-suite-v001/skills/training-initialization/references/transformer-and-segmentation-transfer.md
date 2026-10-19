# Transformer and segmentation transfer: components before recipes

Read when the run uses a Transformer backbone, dense prediction, transferred features, or a model-family recipe whose relevance is uncertain. The useful transfer is a relationship between components, data and optimization; a published learning rate is only one observation under that relationship.

## Contents

- Identify which architecture and input the recipe actually describes.
- Relate component readiness to parameter groups and initialization.
- Diagnose resolution, context and dense supervision.
- Interpret DAFormer and MIC mechanisms without importing their assumptions.
- Work through a transfer decision and its possible responses.

## Architecture changes the training problem

|Component or operation|What it changes|Consequence for the decision|
|Swin patch embedding|A learned strided projection turns image patches into tokens; the reviewed classification implementation expects its configured image dimensions.|Check channel meaning, normalization, physical resolution and checkpoint compatibility before attributing a bad start to the optimizer. A successful tensor load does not establish input compatibility.|
|Swin window and shifted-window blocks|Attention is local within a window in one block; alternating shifts connect neighboring windows across blocks.|A single block does not see every part of the image. A crop or window change alters context availability, not merely batch size.|
|Swin patch merging|Successive stages reduce spatial resolution and increase channel dimensions.|A classification feature can discard details needed by a dense decoder. Inspect the actual feature pyramid and prediction stride.|
|MiT encoder in DAFormer|Overlapping patch embeddings and reduced-sequence attention produce multi-scale features.|MiT is not Swin. Swin window, position-bias and optimizer-branch assumptions do not automatically describe MiT.|
|Classification pooling and head|Spatial features are aggregated into one decision.|A classification checkpoint can initialize useful features without supplying a trained segmentation decoder.|
|Multi-scale decoder|Features at different resolutions are projected, resized and fused into dense predictions.|New projections and fusion layers may need learning while a useful backbone needs protection. An inherited trained decoder has a different starting state.|
|LayerNorm, dropout and stochastic depth|Normalization acts on declared feature dimensions; stochastic layers change train-time computation.|LayerNorm has no BatchNorm running-statistic mixture, but freezing weights alone does not disable stochastic layers. Inspect mode as well as gradients.|

For N tokens and a window containing m tokens, the attention interaction work is proportional to N*m rather than global N squared, holding feature width fixed; projection work also depends on width. Doubling both image dimensions roughly quadruples N. This explains why a higher-resolution run can have very different memory, throughput and usable batch even with the same named backbone. It does not promise fourfold total runtime: kernels, padding, projections and decoder work matter.

The reviewed Swin implementation constructs relative-position indices and shifted-window masks for its configured geometry. Other implementations support dynamic padding or different resolution handling. Identify the actual implementation rather than turning one repository's shape assertion into a Transformer law. Relative-position bias represents offsets within attention; it is not a geographical crop prior.

## Choose groups from readiness and compatibility

Use [functional starts and transfer](functional-starts-and-transfer.md) for initializer and head-bias mechanisms and [normalization and domain mixtures](normalization-freezing-and-domain-mixtures.md) for mode/buffer policies.

1. **Useful backbone, genuinely new dense head.** A lower backbone rate and a more responsive head can reduce damage from initially uninformative head gradients while allowing the head to fit the feature interface. Short head preparation can distinguish feature usefulness from a bad full-model start when that preparation is executable. It is less attractive when target labels are unavailable and only source fitting is observable: a better source head can still make target adaptation worse.
2. **Backbone and head already trained for compatible labels.** Preserve the inherited predictor as the reference. A random-head multiplier or bias reset can erase useful boundaries and calibration. If a particular class regresses, locate its error and exposure before assuming the whole head needs faster learning.
3. **Changed channels, features or input semantics.** A stem may be newly initialized or semantically mismatched even if deeper blocks load. Strongly protecting that stem can prevent adaptation. RGB color statistics, positional geometry and pretrained object semantics need separate justification for multispectral imagery or AEF feature inputs.
4. **Different depth within a transferred backbone.** Layerwise rate decay can protect earlier layers while allowing later ones to adapt when earlier representations remain useful. If the input interface is the main mismatch, that ordering may be inappropriate. Check actual named groups, membership and scheduler handling of multipliers; the configuration field alone proves neither activation nor intended behavior.
5. **Too little task signal to justify broad movement.** Freezing a useful component or using a small adaptation rate can be a reasonable starting choice. It trades representational flexibility for preservation. Stable source metrics do not demonstrate that the frozen features can distinguish the target confusions.

The DAFormer GTA-to-Cityscapes paper uses AdamW with encoder rate 6e-5, decoder rate 6e-4, weight decay 0.01, warmup and a fixed iteration horizon. This illustrates differential readiness; it does not establish a universal tenfold head multiplier. Its reviewed configuration also excludes named normalization/position blocks from decay. In the reviewed Swin optimizer, ordinary training uses decay/no-decay grouping; layerwise rate decay is activated in a specific SimMIM fine-tuning branch. Do not infer layerwise decay from the repository's general support for it.

Record the resolved groups after creation. A name-based rule can miss a renamed module or accidentally cover unrelated parameters. Compare actual changes in the relevant groups with the intended distinction; gradients alone do not establish AdamW displacement. Exact continuation retains the existing optimizer and schedule unless a new intervention is explicitly adopted.

## Dense supervision: a large image is not many independent labels

The loss is applied after a particular resize, crop, ignore mask and reduction. A class present in the full raster may contribute no supervised pixels in the actual crop. A large number of adjacent pixels can supply little independent geographical evidence. These observations imply different responses:

- If a rare class is repeatedly absent from valid crops, adjust permitted sampling or cropping before increasing its loss weight. Weighting cannot amplify a missing gradient.
- If a narrow object disappears after downsampling or label interpolation, inspect target alignment and the feature/output stride. A larger learning rate cannot restore a class removed by preprocessing.
- If contextual classes improve but small boundaries degrade, inspect the scale of evidence and decoder fusion. Aggregate mIoU can hide that tradeoff.
- If increasing resolution forces a much smaller batch, separate the extra spatial detail from changed gradient statistics and normalization. Equal epochs do not isolate resolution.
- If every error occurs at a tile edge, inspect padding, overlap and stitching before changing the backbone. A boundary artifact is not sufficient evidence of inadequate representation.

Semantic masks require a label-preserving resize convention. Soft targets, ignored labels and multi-resolution predictions need their own documented treatment; never interpolate integer class IDs as ordinary continuous image values.

## DAFormer: retain the linked mechanisms

The paper's central improvements are not simply “use a Transformer.” Its study combines an encoder/decoder choice with interventions addressing particular failures:

|Mechanism|Problem addressed and useful evidence|What would weaken the case here?|
|Rare Class Sampling|Rare source classes can be learned too late or not at all, then self-training reinforces common classes. Earlier valid rare-class exposure changes that opportunity.|The target class is not rare in actual crops, labels are unreliable, or oversampling reduces needed diversity and medium-frequency classes. Image selection does not guarantee the desired pixel distribution.|
|Thing-class ImageNet feature distance|Some initially useful ImageNet representations are lost while fitting synthetic source labels. A selective feature-distance term anchors supported object regions.|The inherited features are not useful for this class/input, a semantic mask is unavailable, or anchoring prevents necessary adaptation. The paper's “thing” classes are not a rule that cashew must inherit.|
|Warmup|Large early updates can disrupt a useful transferred representation before the new task stabilizes.|The real problem is wrong inputs, missing gradients or too little effective training; more warmup can postpone learning further.|
|Context-aware multi-level decoder|Features at different scales contribute local detail and context under domain shift.|Context is misleading or unavailable at deployment, or a more complex decoder overfits. A stronger source result alone does not identify the better UDA architecture.|

The paper explicitly reports that improved supervised performance need not translate into better UDA performance. Its feature-distance mechanism concerns a particular pretrained semantic reference, not a general instruction to minimize distance to every source feature. The reviewed code and the paper also use different EMA settings in named configurations; report the chosen executable configuration rather than blending their numbers into a nonexistent recipe.

## MIC: context can help and can mislead

MIC trains predictions of a masked view against a teacher's unmasked prediction. The missing local evidence encourages use of context. This is prediction consistency, not a requirement to reconstruct the original image or AEF embedding. It still depends on teacher semantics and appropriate loss weights.

Useful signs include a task where visible surrounding structure predicts the masked content and a teacher with supported task information. Check how much relevant evidence survives masking, how targets are weighted, and how fast the EMA follows the changing student. The paper uses different teacher smoothing for classification/detection and segmentation and reports sensitivity to this choice; copy neither value solely from the task name.

Failure possibilities are substantive. A context shortcut can amplify an incorrect geographic association. Masking too much or in blocks that erase the only relevant evidence can damage learning. The paper's source-versus-target masking studies differ across synthetic-to-real and real-to-real settings. Applying masking everywhere is not a generally stronger version. For learned feature rasters, channel masking and RGB color transforms are separate operations whose semantic validity needs justification.

## Worked transfer decision

Illustration, not an executed experiment: a new dense head on an inherited encoder initially fits source labels slowly. Low-level feature statistics are finite, labels and outputs align, valid rare-class crops occur, and the head receives updates. The useful inherited predictor is unavailable because this head is genuinely new.

- **Candidate explanation:** the head has not learned the feature interface; jointly moving the encoder may destroy useful features before the head can exploit them.
- **Reasonable action:** if controls allow it, prepare the head with the encoder fixed, retaining the buffer/mode policy, then unfreeze under differentiated rates. An existing compatible head or a diagnosed input mismatch would argue for a different action.
- **Expected response:** head fit improves while inherited features remain useful. On unfreezing, inspect the relevant permitted task errors as well as source loss.
- **If head fitting fails:** revisit feature-label alignment, signal/capacity, gradients and valid class exposure. Blindly increasing all rates is poorly targeted.
- **If source fit improves but target confusions expand:** the preparation demonstrated source fit, not target quality. Reconsider domain-specific alignment or supervision rather than declaring initialization solved.

For Cashew, first read the project component and checkpoint evidence. A canonical source checkpoint may already contain trained task components. “Only the encoder is pretrained” must not be inferred from generic transfer-learning descriptions.

## Source support

- [Swin Transformer paper](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.html), architecture and dense-prediction experiments; [reviewed model and optimizer](https://github.com/microsoft/Swin-Transformer/tree/f82860bfb5225915aca09c3227159ee9e1df874d), `models/swin_transformer.py` and `optimizer.py`.
- [DAFormer paper](https://openaccess.thecvf.com/content/CVPR2022/html/Hoyer_DAFormer_Improving_Network_Architectures_and_Training_Strategies_for_Domain-Adaptive_Semantic_CVPR_2022_paper.html), Sections 3.2–3.3 and 4.2–4.5; [reviewed configuration and implementation](https://github.com/lhoyer/DAFormer/tree/e6f6e041e7477b642106c134c1715fc934ccd210), model, dataset and UDA files.
- [MIC paper](https://openaccess.thecvf.com/content/CVPR2023/html/Hoyer_MIC_Masked_Image_Consistency_for_Context-Enhanced_Domain_Adaptation_CVPR_2023_paper.html), method, source/target masking, mask size/ratio and teacher studies, including supplement.

Component readiness, resolution/crop diagnoses and the worked decision are conditional synthesis of these mechanisms. They are not new measured Cashew findings or a default architecture replacement.
