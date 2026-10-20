# Cashew data, corrected labels and evidence roles

Use when preparing a new Cashew packet, interpreting a historical score, or resolving which supervision/prior belongs to the requested cohort. This is project-specific knowledge. It does not authorize training, relabeling, exporting, refitting a threshold or changing a frozen run.

## Keep the learning and evaluation roles separate

|Asset|Role|What it does not automatically permit|
|Canonical Benin2019 E173|Inherited source predictor for the specified fresh-start contract.|Inventing absent optimizer history, resetting an inherited trained head, or treating it as a full target-run continuation.|
|Source training labels and objective|Four-class weighted CE [3,3,10,3], ignore10, under the supplied source data contract.|Replacing fixed source weights with a class-prior estimate or binary weak-label loss.|
|Human supplementary polygons/bank|Supplementary training evidence in the permitted cohort; no human-derived evaluation/test split under the current user policy.|Calling scores on trained annotations validation/test accuracy, ignoring cohort exclusions or converting unmarked pixels into negatives.|
|Agent-selected automatic negatives|Separate low-weight, reversible weak hypotheses under the advertised route.|Calling them human truth, inheriting the old disputed weak-negative pool, or overwriting source labels.|
|Current country/year point validation|Separate development validation; its measured feedback is available to the Agent for tuning, STOP and SELECT under the current user policy.|Putting validation labels in the loss, calling validation-tuned scores independent final tests or silently rewriting a frozen cohort.|
|Fixed unlabeled monitor|Prediction behavior on its exact pinned patches/population.|Positive recall, target F1 or full-tile area without additional support.|
|Approved regional prior|Spatial/agronomic support, confidence and unknown/conflict information.|Calibrated cashew probability, a prevalence quota or pixel truth.|
|Standalone PlanetMap|A separate dated reference product under its own class mapping.|Substitution for the approved regional prior or automatic labeling of every disagreement.|

Ordinary class order is 0 crop_others, 1 mixed_trees_grassland, 2 cashew, 3 built_up; ignore is 10. “Other than cashew” in a binary error analysis can combine 0/1/3, but that does not rename the model's class0. There is no model wetland class.

## Current user policy: human training samples and separate validation feedback

This user amendment supersedes the earlier human-location holdout default for newly prepared or explicitly migrated Cashew cohorts. Keep permitted supplementary human positives and negatives in the training pool. Do not reserve part of that pool as validation/test by splitting points, polygons, patches, pixels or locations. Existing cohort bans on human supervision still apply. Unmarked pixels remain ignored. Do not reclaim material explicitly designated as final test or silently resolve a training/validation overlap by training on it; identify and exclude the overlapping training support under the cohort's declared identity rule.

Use the separately designated, corrected country/year validation points to supply Agent feedback. A worker can evaluate the current checkpoint at scheduled reviews and pass compact per-class F1, precision/recall, FP/FN, support, label version, decoder/threshold and checkpoint identity. The Agent may use these observations for adjustment, STOP and SELECT; an evaluated starting model may also inform initialization. The current user's permission covers feedback; validation labels do not become auxiliary-loss targets or automatic pseudo-negative labels. Report training-sample scores only as training diagnostics. Keep separate final TEST evidence out of decisions; if no independent test exists, report validation performance without claiming an independent test.

When a validation threshold is optimized, distinguish its fitted F1 from fixed-decoder F1 and name the threshold-fitting population. Repeated validation use can overfit the selection process; do not describe the resulting maximum as untouched accuracy. Compact feedback is sufficient unless an authorized diagnosis requires finer error slices; there is no need to dump all point attributes into every request.

Historical packets that used `heldout_locations` from human polygons or set `point_labels_enter_training_or_online_selection=false` retain those facts. For the new protocol, the preparer must remove the artificial human holdout, include the eligible human annotations in training, identify the separate validation dataset, and connect its feedback to review/selection packets. Register the changed data roles and the first affected review in a new version. Editing Skill text alone does not migrate a frozen runtime, retroactively change an F1 interpretation or establish that feedback was delivered.

## Human positives: resolve the cohort, not the latest familiar filename

The approved original positive input for the fresh CIV contract is:

    /mnt/nas_data1/cashew/country_shp/data/supplyment_sample/cashew.kmz

That path exists at this construction check. The requested fresh-CIV source is `cashew.kmz`, not `cashew2.kmz`. Historical runs that pinned a different polygon pool retain their original identity; do not rewrite their results or pretend they trained on this file.

A later user-authorized combined manual bank is resolved by:

    /mnt/nas_data1/cashew/human_sample_banks/current_bank.json

At this check it names `CIV_current_plus_b_manual_20261016_v001` in the same directory. Its scope includes the specified Senegal cohorts and later local jobs. The pointer explicitly excludes human samples for the named Nigeria44 AgentGuide cohort. A global bank containing Nigeria patches does not override that exclusion. The cohort's human count can be zero while its separate automatic-negative route remains available.

Resolve the bank's immutable preparation/verification records and the actual packet's selected locations before describing its active exposure. The pointer is not proof that every bank sample was used. Unmarked supplementary pixels stay ignored; deleting/shrinking a positive polygon does not turn its removed pixels into negatives.

## Current validation and cross-year identity

Canonical roots:

    /mnt/nas_data1/cashew/country_shp/data/2019/
    /mnt/nas_data1/cashew/country_shp/data/2025/
    /mnt/nas_data1/cashew/country_shp/data/sample_change_records/

For an authorized new evaluation, the evaluator/preparer reads the change-record index and current country SHP. A scientific Agent without that label permission receives version/provenance facts and permitted metrics rather than opening the labels. A mirror under `data/sample` is not automatically authoritative; an unaffected older mirror can differ. A Shapefile's DBF carries attributes, so a matching geometry file alone does not establish matching labels.

The latest recorded correction batch at this construction check is:

    /mnt/nas_data1/cashew/country_shp/data/sample_change_records/20261015_2019_2025_point_corrections_v002/

Its `change_record.json`, `changes.csv`, `independent_verification.json` and `current_cross_year_point_id_lookup.csv` retain the operations and current lookup. The record reports 20 geographic anchors and 29 point-year actions. Current canonical components for its 16 affected datasets were rechecked against the recorded post-edit hashes: all 80 recorded components matched. This checks those identities; it is not a fresh independent land-cover/date audit or a new evaluation result.

The preceding ID alignment record is:

    /mnt/nas_data1/cashew/country_shp/data/sample_change_records/20261015_2019_point_ids_aligned_to_2025_v001/

Same locations share the 2025-anchored point_id; original 2019 IDs remain in `orig_pid19`. Originally 2019-only locations received new IDs. Use the latest correction's current lookup, not an older crosswalk containing subsequently deleted rows, to decide whether a point still exists.

The TGO same-coordinate duplicate has a final user resolution that supersedes the earlier provisional instruction. Its year-specific deletion/retention is preserved in the change records and latest lookup; do not recreate deleted rows from an older table. Point coordinates and corrected target classes stay in those permission-controlled data records, not in this general decision prompt. The lookup includes class columns and therefore is not a label-free identity service for a sealed initializer.

Both years of a spatial tile belong in the same research partition and both must be excluded from held-tile experience where that partition is sealed. Two years are useful response contexts, not independent geography. Existing numerical train/monitor lists and source E173 were not changed by these validation corrections. Old scores remain attached to the labels actually used; a new score needs the current permitted labels and its own provenance.

## Prior product identity

Authoritative regional release:

    /mnt/nas_data1/cashew/experiments/WestAfrica_regional_cashew_prior_20261014_v002/manifest.json

Resolve countries through its `country_results`. Some v2 entries intentionally reuse v1 immutable raster bytes. The release covers BEN/BFA/CIV/GHA/GIN/GMB/GNB/MLI/NGA/SEN/SLE/TGO; inclusion in the prior does not imply inclusion in a current training experiment.

CIV keeps the approved v3 raster:

    /mnt/nas_data1/cashew/experiments/CIV_regional_cashew_prior_20261014_v003/CIV_regional_prior.tif

Recorded SHA-256: `7bd65d5416758d394a5390e427c0157bb9aea49e218f26dafb1816307203c2fc`.

The regional product is distinct from standalone 2017 PlanetMap. Its common raw-score scale can help compare stronger support across WestAfrica, while evidence_confidence, text_unknown and evidence_conflict limit interpretation. Country ranks or per-tile quantiles cannot silently replace that common scale. In the registered grid, equations are EPSG:6933/1km-based although historical WKT can appear as LOCAL_CS; use the manifest/transform rather than treating coordinates as degrees.

The user excluded Mali from the present experimental concern. Preserve any existing prior and historical record, but do not add a new Mali training obligation. Prior availability itself creates no launch permission.

## Current grouping exclusion and historical contracts

The user subsequently excluded CIV groups8/9/10 of the current formal all-CIV10 grouping from future launches, retries and continuations, for both years and AgentGuide/fixed. Resolve exact memberships from:

    /mnt/nas_data1/cashew/fada_expert_runs/training_scope_policy.json

Completed artifacts are retained. Group numbers are not transferable identities for a different clustering. The earlier ten-group fresh-run description remains historical context and does not override this explicit later scope policy.

Other historical cohort rules, including minimum training, accounts, human routes and schedule phases, stay with their pinned packets. Do not turn a production cohort's epoch floor or proxy route into a general training-expert rule. The new suite permits explicitly authorized small real-data CPU trials through its isolated adapter; existing production processes remain outside this upgrade's scope.

The prior baseline for this upgrade is validation-point feedback during AgentGuide. A formal threshold study or systematic method comparison had not been performed for this new protocol; do not rewrite historical studies with different roles into that claim. New feedback uses fixed multiclass argmax and a finite development-use ledger. Validation guides diagnosis, candidate methods and bounded experiments, while labels remain outside backpropagation. New independent evaluation, threshold optimization and formal comparisons need their own prospective task contract.

## Source and maintenance

These are user/project contracts verified against the named files during construction, not literature-derived prescriptions. When a pointer changes, retain the original frozen input identity and resolve the new cohort explicitly. Update the smallest affected fact and its provenance rather than silently changing historical metrics. Use the general state/metric references to interpret what each record supports.
