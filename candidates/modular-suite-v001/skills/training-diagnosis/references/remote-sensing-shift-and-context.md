# Remote-sensing shift, physical context and representation

Read when transferring a published remote-sensing method, diagnosing a geographic error pattern, or deciding whether the problem concerns input appearance, physical context, label semantics or optimization. Similar network names do not make two image domains or tasks equivalent.

## Identify the level at which information differs

|Difference|What can change|A targeted interpretation|
|Sensor/channel semantics|A channel can represent a different wavelength or learned quantity; normalization can alter its scale.|Check the actual input meaning and preprocessing before attributing failure to LR. A tensor with the expected shape can still be semantically incompatible.|
|Ground sampling distance and crop size|The same pixel window can cover very different physical objects and context.|Reason in ground units as well as pixels. Increasing context and increasing detail are different interventions.|
|Acquisition date, season and land-cover change|Appearance and the class itself may change.|Separate temporal appearance shift from a real label change. Cross-year points need spatial and label-version identities.|
|Task taxonomy|Binary farmland, orchard canopy, cashew plantation and mixed woody cover have different boundaries.|A high farmland recall is not evidence of cashew precision or separation from other trees.|
|Region and sampling|Class mixtures, neighboring objects and the proportion of nearly pure background tiles can differ.|A curated mixed-class patch benchmark may not represent deployment in a nearly absent-class group.|
|Output/stitching geometry|Stride, resizing, padding, overlap and valid masks affect local decisions.|Edge artifacts and disappearing narrow objects can be geometric failures rather than optimization failures.|

A larger input-feature distance may reflect any combination of these differences and model sensitivity. It does not identify which semantic class changed or prescribe one learning rate. Condition the comparison on relevant classes or supported errors when available; a pooled distance can be driven by a changing background mixture.

## Physical context can explain a task-specific input choice

A 512-pixel crop at 0.3, 0.5 and 4 metres per pixel spans about 154, 256 and 2048 metres per side. These are the resolutions encountered in the agricultural-land study, not the current AEF resolution. A fixed window can therefore show part of an object in one domain and a large landscape in another.

When a class needs contextual evidence, increasing crop extent may help. If the object is lost through downsampling, greater spatial detail or a suitable decoder may be more relevant. Larger crops can force smaller batches and change optimization statistics. Inspect which quantity actually changed: number of supervised pixels, physical extent, effective batch, valid class exposure or merely tensor size.

The agricultural study chose a crop size with a contextual motivation but did not establish it as a universally optimal size through a fully matched resolution/crop experiment. Use the mechanism to choose a question for the current data; do not reuse 512 simply because both projects use a Swin backbone.

## Input-style alignment: what Fourier transfer assumes

The user-supplied DSP2024 paper changes input appearance by replacing a low-frequency amplitude region of a source image's Fourier transform with the target amplitude while retaining the source phase. In symbolic form:

    amplitude_mix = M_beta * amplitude_target
                    + (1-M_beta) * amplitude_source
    styled_source = inverse_Fourier(amplitude_mix * exp(i*phase_source))

The purpose is to change appearance while retaining label-relevant content. Keeping phase is a motivation and an empirical assumption, not a theorem that arbitrary amplitude replacement preserves every semantic label. Large transformations can create artifacts or alter useful evidence.

The paper allows different beta values by source semantic class. It generates candidate styled images and composes them using dense source label masks. Its premise is that vegetation appearance can react differently to the RGB/IRRG sensor change than some other categories. This is a concrete example of selecting an intervention according to class behavior rather than applying one scalar to the whole domain.

It nevertheless uses experimentally selected beta values under its validation protocol. “Class-aware” or “adaptive” does not mean an online Agent automatically inferred those values without target feedback. Sparse point labels do not provide the same dense composition mask. AEF's 64 learned dimensions are not RGB/IRRG optical channels; applying the same operation to embeddings would be a distinct method with unestablished label-preservation assumptions.

If the current problem is physical channel appearance and the necessary source labels/input data are available, input-style alignment can be a justified research candidate under a supporting contract. If the problem is orchard-versus-forest semantic confusion in a learned representation, changing raw-image style may not address it. Increasing adversarial strength cannot repair missing sensor information either.

## Extra objectives change exposure as well as method

The DSP method supervises both original and styled source inputs and uses two fine-grained discriminators: original-source versus target, and styled-source versus target. Those roles are not the same as a particular project's local/global discriminators. Each loss must retain its own active path, normalization and sample exposure.

Its ablations contain a useful counterexample: in Potsdam-to-Vaihingen, adding one adversarial term to the two supervised terms reduced mIoU from 38.20 to 37.50 while overall pixel accuracy rose from 63.22 to 63.81. This is the paper's report, not a Cashew result. It shows why an apparently improved aggregate does not automatically validate an auxiliary term for the declared objective. The full combination's result does not prove every intermediate addition was beneficial.

If copying a method with more supervised views, count the extra fitting and data transformations. A comparison to an uncontinued source checkpoint does not isolate adversarial benefit. An equal-epoch comparison can still differ in actual forwards, optimizers and total useful-label exposure.

## Within-target heterogeneity and entropy

The agricultural Swin study addresses differences within the target through an entropy-based easy/hard split followed by another adaptation stage. This is a useful hypothesis when one country or tile group contains different error regimes. It is not proof that low entropy identifies correct positives or negatives. A model confidently predicting one class everywhere can have low entropy.

For an actual split, inspect the score's sign, reduction, ordering and conditioning. The reviewed public snapshot sorts positive entropy in descending order before naming the first subset “easy,” opposite the paper's low-entropy explanation. This is a version-specific implementation observation; it does not establish which implementation produced the published results. The useful lesson is to recover the real selection rule before interpreting a named subgroup.

A class-conditional or region-specific failure can be obscured by one global domain score. Use relevant error slices and context, but do not create unsupported geographic quotas or hard absence masks. Shared evidence from one model's score, entropy and pseudo-label frequency is not three independent semantic measurements.

## Interpret agricultural metrics according to the actual product

In that binary agricultural study, COM is recall and COR is precision. A reported high F1 can combine a recall gain with a precision decline. One direction reports final recall about 92.1% and precision about 56.9%. This does not demonstrate that a rare-cashew overprediction problem has been solved.

The reviewed evaluation code also conditionally includes images in some per-image averages only when positive true predictions and positive support occur. Global confusion accumulation is a different statistic. A mean that omits entirely missed positives or false positives on all-negative images cannot establish performance on those cases. Do not use the algebraic F1/IoU relationship on differently aggregated metrics to infer a paper's calculation error; first recover the populations and aggregation.

Patch curation matters as much as the formula. The paper describes excluding patches with insufficient agricultural/background pixels; its resulting mixed-class benchmark does not establish performance on nearly pure background tiles. Likewise random patch splits do not by themselves demonstrate independent geographical evaluation. These are scope limits, not proof of leaked pixels in an unaudited dataset.

For a map whose main concern is excessive cashew, retain supported FP behavior alongside recall/F1 and distinguish point-panel accuracy, canopy area and plantation footprint. A pleasing dense output or a higher aggregate metric cannot replace that assessment.

## A useful expert response

Illustration: two domains share a model architecture and input shape, but one has a very different ground resolution and the errors cluster around small orchards. Check physical support, resizing and output stride first. If geometry is correct and classes are exposed, inspect whether the inherited features separate orchards from the confusing background. Choose a permitted scale/component/optimization treatment that addresses the supported bottleneck. A generic “increase adaptation because feature distance is large” would not distinguish these cases.

For Cashew, combine this knowledge with the supplied AEF contract, real prior scope and component paths. Do not claim access to raw spectral, radar, texture or seasonal inputs that the packet does not contain. Regional agronomic context can guide a hypothesis while remaining distinct from a human label.

## Sources and limits

- Ismael, Kayabol and Aptoula, [Unsupervised domain adaptation for the semantic segmentation of remote sensing images via a class-aware Fourier transform and a fine-grained discriminator](https://doi.org/10.1016/j.dsp.2024.104551), Sections3–4 and ablations. User-supplied PDF is archived as S040; no author implementation was inspected for that source.
- Zhang et al., [Unsupervised Adversarial Domain Adaptation for Agricultural Land Extraction of Remote Sensing Images](https://doi.org/10.3390/rs14246298), publisher v2, input/data setup, two stages, metrics and experiments; [reviewed code](https://github.com/ZJunBo/agricultural-land-extraction/tree/79581140549b65e46e384d7623617a5e155a85d1), selected entropy, training, model and evaluation paths. Reviewed files are recorded under S046; whole-repository execution was not performed.

The cross-task decision example is conditional synthesis, not an executed treatment or a claim that either recipe improves AEF Cashew training. The inaccessible S047 paper was explicitly waived by the user and supplies no knowledge claims here.
