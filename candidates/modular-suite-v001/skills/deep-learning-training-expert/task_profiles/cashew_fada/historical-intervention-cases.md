# Historical Cashew interventions: repeated responses and their limits

Use the multi-tile evidence below before retrieving an illustrative single case. The four original examples at the end are local observations, not independently established tuning rules. The 2026-10-19 expansion read historical records only: no training, GPU work, scientific model calls, new label access or threshold fitting. It did not change the original frozen decisions or evaluations.

## Scope, matching and evidence strength

The previous 200-directory convenience sample was insufficient coverage. This review searched comparison/summary/experience files in `fada_expert_runs`, the Skill-refactor evaluation and initialization-study trees, and followed all 727 logbook documents rather than stopping at the first 200 experiments. It read 2,096 `comparison.json` files and retained 1,479 original, completed, recorded single-control comparisons with a resolved single global tile and compatible full-list populations. These cover 239 distinct global tile IDs and eight changed-control names. Twelve linked experiment directories had no comparison. Remaining records include group-weighted estimates, compound changes and other schemas; exclusion from this single-tile analysis does not mean their training failed. This is coverage of the searched local archive, not every historical server or a population success rate.

The analysis keeps exact action values, completed update budget, starting epoch, target year, recorded source profile, source validation support and background controls separate. It identifies 35 recorded-context strata with at least three tiles. A second, separately labelled matching level permits intentionally different tile-specific pools under the same selection procedure (53 such strata); it records pool variation rather than pretending the complete starting states match. These are exploratory strata, not predeclared effect tests. Neighboring tiles and shared source models can remain correlated.

Years, seeds, reviews and branches of the same global tile are repeated observations, not extra spatial replications. A group containing ten tiles is not ten independent training experiments. A per-tile median summarizes repeated branch responses; positive/negative counts below refer to distinct tiles. For the fixed 192-run quality panel, preserve its original mean over paired year/seed cells within tile instead.

The compact conclusions are here. [Numerical evidence and original artifact identities](historical-multitile-evidence.json) are for checking a claim, not routine prompt loading. Original checkpoints/RNG/runtime bytes were not revalidated by reading comparison JSON. Matching recorded controls and populations does not resolve all historical execution confounding.

## Repeated responses that can inform a decision

All differences are treatment minus unchanged control; percentage points are fractions multiplied by 100. Map fractions are four-class argmax class-2 valid-pixel fractions, not thresholded p2 or accuracy. Source F1 and target quality stay separate.

|Context and executed action|Cross-tile response|Permitted use|
|---|---|---|
|Recorded `source2025_ghana_e173`, target 2025, epoch 5, 32 updates; lambda 0.004 to 0.002; model LR 2e-6, D LR 5e-6, two D steps; shared historical 413-pair supplement|43/43 tiles had a lower effective gradient ratio. Target class-2 fraction was lower on 34 and higher on 9; source class-2 F1 fell on 32 and rose on 11.|Repeated pressure control, with heterogeneous area and source responses. Do not promote it to reliable correction of false positives or target accuracy.|
|Same recorded source/year/epoch/budget; D LR 5e-6 to 2e-6, lambda 0.004, model LR 2e-6, two D steps|41/41 tiles had lower local D accuracy and lower effective gradient ratio. Target class-2 fraction fell on 26 and rose on 15.|D rate can change opponent/gradient behavior in this regime. Neither lower D accuracy nor lower pressure ranks target quality.|
|Recorded `source2019_benin_e173`, target 2025, epoch 5, 32 updates; lambda 0.004 to 0.002, both LRs 2e-6; same historical tile-pool selection procedure, supplementary weight zero|6/6 tiles had a lower ratio; target class-2 fraction fell on 5 and rose on 1; source F1 split 3/3. Across the broader 78-tile same-action/budget archive with other contexts varying, area signs split 42 lower / 36 higher by tile median.|The mechanical response repeats in a later recorded source protocol. The broader record prevents treating a favorable narrow stratum as a universal area recipe.|
|Recorded `source2019_benin_e173`, target 2025, epoch 5, 315 updates; D LR 5e-6 to 2e-6, model LR 2e-6, lambda 0.004; supplementary weight zero|5/5 tiles had lower target class-2 fraction (median difference -0.599 percentage points). Source F1 fell on 4 and rose on 1.|A scoped, repeated area response; it does not establish that 315 updates, lower D rate or less area improves the target. Other epochs, durations and opponent states need their own evidence.|

These records use historical supplementary policies, not the present automatic-negative mechanism. Recorded profile names and checkpoint paths are both retained; do not infer a checkpoint's training country from its directory name or silently merge source protocols. None of these scalar values becomes a production default.

## Revision: lower model LR is not a general quiet-map rule

Case B below showed a quieter map on one tile. Expanding its exact action, model LR 5e-6 to 2e-6 for 32 updates, finds 11 comparisons on ten tiles with other contexts varying: tile-median turnover fell on eight and rose on two; source class-2 F1 improved on five and worsened on five. Retain this majority motion trend and its failures, not a uniform accuracy prescription.

A different, internally matched recorded 2025/GHA-profile stratum at epoch 25 compares model LR 2e-6 to 1e-6 for 32 updates, holding D LR 2e-6, lambda 0.004, two D steps, temperature 1.8, cap 0.9 and both decays 0.01 fixed, with the shared 413-pair supplement. On all four distinct tiles (`18`, `9d`, `185`, `21b`; Nigeria/Ghana):

- Effective gradient ratio increased and target class-2 fraction decreased.
- Class-2 turnover **increased** and temporal class-2 IoU to the starting map **decreased**.
- Source class-2 F1 improved on two and worsened on two.

Median extra turnover was 0.687 percentage points; median temporal-IoU difference was -0.0519. Turnover means `(gained + lost) / valid_pixels`, not the signed area change. A map can lose more cashew and move farther from its start simultaneously.

The revised knowledge is the observed interaction and uncertainty, not an opposite rule that lower LR always destabilizes maps. Check the existing optimizer/opponent state and actual trajectory; do not infer stability or accuracy from the LR direction. The old single-tile preservation response remains a case-level hypothesis.

## Weak-negative activation: a repeated local response with real counterexamples

Four tiles (`b0`, `2a2`, `1c8`, `a6`; Ghana/Ivoire/Gambia) have comparable recorded Benin-profile conditions at epoch 60: 315 updates per arm, both LRs 2e-6, lambda 0.004, two D steps, cap 0.9, temperature 1.8. Changing historical `supplementary_loss_weight` from 0 to 0.1 used nonzero selected negative pixels on every treatment arm. Pools differed intentionally (133–533 pairs) under the same historical geographic/AEF selection procedure and one-occurrence control.

All four showed lower full-target mean p2, lower class-2 fraction and higher **source** class-2 F1. Median differences were -0.0127 p2, -1.722 percentage points of predicted class-2 fraction and +0.633 source-F1 percentage points. This supports a repeated response in this scoped stratum, not a target-accuracy win.

The broader same-source, same-action, same-315-update archive contains **23 distinct tiles** with other controls/stages varying: class-2 fraction and mean p2 fell on 15 but rose on 8; source F1 improved on 11 and worsened on 12. All 23 treatment arms recorded positive selected-pixel exposure, so the opposite responses cannot simply be dismissed as an empty negative pool. Recorded cumulative selected negative pixels ranged from 16,363 to 8,582,451 (these are exposures, not unique pixels). The same weight is therefore not the same supervision dose.

Use the experience to inspect effective exposure, normalization, competing updates and whether actual errors support negative supervision. Retain the contrary cases; do not fit a universal pool-size cutoff after seeing outcomes. The epoch-60 stratum was found retrospectively and does not justify a mandatory activation epoch. Its old weak-label bank and weight 0.1 are **not** the current Agent-selected rising-p2/prior mechanism or its permitted weight scale. Do not copy the historical geographic filter, activation restriction or scalar into that mechanism.

## D cadence: use the real multi-tile quality result

The existing frozen 192-run study supplies a separate quality check. Among the original C-expert choices, 30 paired year/seed cells used one D step while their fixed reference used two; the other five initial scalar controls matched. At the predeclared E10 endpoint, these cover **eight spatial tiles**, not 30 independent tasks. Averaging the original paired cells within each tile gives four positive class-2 F1 differences, three equal and one negative. Two of the four positive-mean tiles also contain adverse cells. These selected action subsets are descriptive, not randomized evidence that one D step is optimal.

The full original study and uncertainty stay unchanged: no established stable superiority, and no hindsight rewrite of its decisions. The useful update is to retain heterogeneous task-quality feedback rather than extrapolate the two early DEV successes into a universal one-D policy. A D-only work reduction is a separate outcome. These disclosed outcomes are now historical knowledge and cannot later serve as unseen validation of this revised Skill.

## What remains an observation or an unconfirmed hypothesis

Temperature, decay and other archive controls lack enough comparable multi-tile evidence here for a useful quality recommendation. Small short-horizon changes, mixed signs or a uniform tiny change are not evidence of equivalence. A repeated mechanical pattern may guide diagnosis; a quality prescription needs repeated permitted quality evidence and its failures. No new training is authorized or required by this historical knowledge update.

## Original single-tile illustrations, retained without promotion

The following cases preserve the original numbers and hashes. They are older 2019 runs, not evaluations under the latest corrected target labels. Each records 32 steps per arm and no main-checkpoint replacement. Later main-run adoption is a separate event. Read them as local observations under the multi-tile revision above.

## Case A: lower lambda, more expansion

Tile `Tile2019_00000000000000000003`, start epoch 5, experiment `94f2460b0a0e`:

|Quantity|Control|Treatment|
|---|---:|---:|
|lambda_adv|0.004|0.001|
|Recorded effective gradient ratio|0.15885479|0.04842820|
|Class-2 fraction change from common start|0.00959240|0.01577267|
|Source class-2 F1|0.78294177|0.78199360|

The lower coefficient reduced measured weighted pressure, yet treatment expanded the predicted class-2 fraction by an additional 0.00618028, or about 0.618 percentage points, relative to control. This is a direct counterexample to treating lambda as a monotone cashew-area knob. It supports the mechanical coefficient effect while weakening the proposed expansion-control explanation at this state. Neither area change establishes target accuracy.

For a similar current case, inspect the semantic path, competing gradients, score scale and error support. Do not repeatedly lower lambda merely because its measured ratio reliably falls.

## Case B: lower main LR, quieter map, higher ratio

Same tile, start epoch 15, experiment `17bce79a1257`:

|Quantity|Control|Treatment|
|---|---:|---:|
|Main LR|5e-6|2e-6|
|Recorded effective gradient ratio|0.13410393|0.18077644|
|Class-2 fraction change from common start|0.03010460|0.00490886|
|Temporal class-2 IoU to common start|0.58701320|0.73694310|
|Source class-2 F1|0.80838588|0.81265468|

The treatment reduced expansion by about 2.520 percentage points relative to control, retained more of the starting map and improved this source F1 by about 0.427 percentage points. Meanwhile the gradient ratio rose. The useful lesson is that model motion, the ratio and source quality are distinct quantities. This supplies a local source/motion response, not measured target improvement or a universal LR recommendation.

For a similar case, lower model rate remains a possible local hypothesis. The four-tile contrary response above prevents promoting this illustration into a general preservation rule. Check whether a quieter map preserves useful target positives or inhibits needed adaptation; unlabeled stability cannot answer that alone.

## Case C: cadence is a separate intervention

Same tile, start epoch 20, experiment `adc3d34ebbff`: D steps per model step changed from 2 to 1 while D LR remained 5e-6 and model LR 2e-6. The recorded ratio changed from 0.19156921 to 0.18439886; class-2 fraction change from start changed from 0.01649384 to 0.00641230. Source class-2 F1 was 0.81541412 versus 0.81572150.

The recorded response supports a cadence-dependent mechanical/motion difference in this setting. It does not identify D rate as the cause or show a meaningful target-quality gain. Changing both cadence and D LR in the new contrast would ask a different question. Count actual D work and its buffer/data effects rather than equating cadence with a rate multiplier.

## Case D: temperature response was tiny

Tile `Tile2019_00000000000000000014`, start epoch 60, experiment `b04e4dc34271`: temperature changed from 1.8 to 2.2. The treatment-minus-control class-2 fraction change was about 0.00000230, only about 0.000230 percentage points. Ratio difference was about -0.00005412; source class-2 F1 was 0.81884152 versus 0.81883172.

This short comparison gives little support to a consequential temperature intervention at this state. It does not establish equivalence at all horizons, nor that temperature is never useful. Inspect whether conditioning actually changed and whether the observed noise/exposure can resolve the question. Retain the null-like case alongside more visually impressive interventions.

## A data-version counterexample

The historical epoch-15 request for tile `...0003` recorded a 413-pair supplementary list under `20260906/epoch173_source1100_supplementary_segonly_v4_soft_cap_09/data/supplementary_segmentation_only.txt`. A later job file pointed to 1,012 pairs under `20260916_supplementary_all_1012/data/supplementary_segmentation_only_all_1012.txt`.

Reusing the directory or epoch without the request's original data identity would change the experiment. These old supplementary routes are not the present `cashew.kmz` contract or the latest corrected cross-year evaluation set. Preserve the local mechanism observation while restricting any transfer of quality claims.

## Original artifact identities

All paths are rooted at:

    /mnt/nas_data1/cashew/fada_expert_runs/planet_class2_over_1pct_tile_only_two_phase_20260910/2019/

|Case|Relative comparison path|SHA-256|
|---|---|---|
|A|`Tile2019_00000000000000000003/experiments/epoch_005/94f2460b0a0e/comparison.json`|`2a6dfc8875b056ba56394b1467468c62611f420982ee1e5330a04ed7a591d84c`|
|B|`Tile2019_00000000000000000003/experiments/epoch_015/17bce79a1257/comparison.json`|`b5fcd3d5024f153abb3a866ea150a45aba4f6a6fa8f876877308f1864e46d436`|
|C|`Tile2019_00000000000000000003/experiments/epoch_020/adc3d34ebbff/comparison.json`|`330b4cd8f9f6a5f9e3bd84dd32158c055ad7f282dda053b31056a674ba579d32`|
|D|`Tile2019_00000000000000000014/experiments/epoch_060/b04e4dc34271/comparison.json`|`20e81865874b317544d5b0170d99139b1ae58bfca7f22ed4c1d1d01d46b93fff`|

The bounded sampling record and earlier interpretations are in the maintained historical Skill's `references/history-evidence.json` and `references/historical-lessons.md`. Source quality, map motion and semantic conclusions remain separate even when the arithmetic is verified.
