# Historical Cashew interventions: what the responses actually establish

Use these cases when the mechanism resembles the current question. They are older 2019 runs, not results of the present knowledge-construction candidate, and not evaluations under the latest corrected validation labels. Four original comparison files were read again during construction; their current SHA-256 values match the retained history evidence. No training was rerun.

## Scope of the historical sample

The earlier index contained 727 job entries. Its bounded convenience sample followed the first 200 explicitly linked `two_phase` experiment directories in index order: 196 recorded completed comparisons and four without a completed comparison. Those counts are neither an unbiased success rate nor evidence of all jobs' scientific validity. A manifest saying `running` is not a live-process check.

The cases below each record 32 model steps per arm, a single changed control and no main-checkpoint replacement by the branch executor. The comparison contains useful source metrics and full-list unlabeled prediction diagnostics. It does not by itself prove matching historical checkpoint bytes, RNG/order, all implementation versions or target accuracy. Later adoption in a main run is a separate event.

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

For a similar case, a lower model rate can be a justified preservation intervention. Check whether it preserves useful target positives or merely inhibits needed adaptation; unlabeled map stability cannot answer that alone.

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
