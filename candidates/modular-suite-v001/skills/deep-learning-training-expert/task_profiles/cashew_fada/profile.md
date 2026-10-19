# Cashew FADA project profile

Apply only for a supplied Cashew FADA contract. A historical packet keeps its
own source/year, policy, label route, decoder and bounds. This profile does not
migrate frozen runs. The worker, not the decision model, owns GPU admission.

When sample version, human/automatic supervision, corrected cross-year IDs or prior identity matters, read [data, labels and evidence roles](data-labels-and-evidence-roles.md). It records actual project paths, correction lineage, cohort-specific human routes and the later CIV8/9/10 scope exclusion. Historical ten-group descriptions below do not override that later user policy.

## Model and metric identity

The project order is 0 crop_others, 1 mixed_trees_grassland, 2 cashew, 3 built_up;
10 is ignored. Cashew is index2. Named metrics must agree with their source
arrays; do not use a Class3 value to infer Cashew collapse. Ordinary current
delivery is four-class argmax==2; p2 and fitted thresholds are different outputs.
Historical outputs may follow a different pinned decoder.

Canonical source is Benin2019 E173. The current source objective is weighted
four-class CE [3,3,10,3], unchanged. Do not replace it with a binary helper or
retune the class weights. In the audited producer, training `cls` is the
arithmetic mean of per-step weighted CE values normalized by the sum of valid
label weights. Source-validation `loss` instead averages per-batch weighted CE
normalized by valid pixel count, then gives each batch scalar equal weight.
Their scales depend on class mixture and batch support; they cannot be directly
subtracted as a train/validation gap or converted by one universal multiplier.
This naming difference does not alter the class weights or the separate F1
counts. Verify another producer's definition before applying this interpretation.
Current A3 E0 inherits all217 E173 state tensors,
including decoder/fuse and the encoder; new residual outlets initially preserve
the inherited function. New FADA discriminators and optimizers are separate.
Legacy DA layers can be inactive. Verify the supplied component audit rather
than assuming these facts for another architecture.

The currently audited canonical E173 file is a bare217-tensor state mapping,
without saved optimizer history; the declared epoch173 comes from its pinned
source provenance. The audited A3 runtime has12 new A3 state tensors; the8
legacy `fc*_DA`/root `norm` tensors need path-specific interpretation. Classification
hooks skip all8. FADA Stage4 extraction calls root `norm`, however, and its
parameters receive adversarial gradients and model-optimizer updates. The6
`fc*_DA` tensors remain unused in these audited objectives. See
[component loss paths](component-loss-paths.md) when component activity matters.
A loading or forward-hook result does not establish historical training of every tensor.

For concrete tile-conditioned choices, read [tile evidence to training choices](tile-evidence-to-training-choices.md). It relates inherited behavior, valid class exposure, opponent readiness, woody confusion, prior context and schedule to different actions. [Historical intervention cases](historical-intervention-cases.md) preserve actual useful, contradictory and nearly null responses; they are not current-label target-quality wins.

AEF has64 embedding dimensions, not64 optical spectral bands. Source/target
AEF distance and class competition can guide hypotheses; neither identifies
target truth. Clipped softmax(logits/T) used for conditioning is not normalized
again in the current worker. T is not a delivery threshold. Lambda affects the
model adversarial term; D learning rate/cadence affects the signal-producing D.
Measured weighted gradient ratios are not AdamW update ratios.
In the current pre-run gradient table, `lambda_candidates` keys name desired
weighted gradient ratios and values are lambda amplitudes; verify the forward
estimate `lambda * raw_adv_to_segmentation_ratio`. That is a probe estimate,
not an observed optimizer response. The profile's [packet contract helper](packet_contract.py)
can expose this arithmetic, actual control names and probe split sizes without
choosing a scientific action. Supply its output to both arms of a comparison.

For discriminator pre-run step labels and prepared-state comparisons, read
[initialization diagnostic semantics](initialization-probe-semantics.md). Apply
its legacy timing interpretation only with the matching producer identity; newer
measurements can report completed updates directly. Cached D fitting, out-of-fit
domain prediction and target class quality remain different evidence.

## Supervision and regional context

Follow the packet's authorized label route. Keep source validation, supplementary
training labels, held-out human error panels, TEST labels and Planet summaries
distinct. Labels excluded from the decision must remain excluded. Current
corrected cross-year validation IDs are anchored to2025; both years of one global
tile stay in one research partition. Do not read a TEST reference panel to
construct a new initializer.

When supplied, use the packet's `observation_population` to interpret current
predictions. A conditional512-patch DEV monitor is neither the production CIV3000
sample nor full-tile area. The approved prior row's training/evaluation sample
counts describe its original sampling; they do not become the current monitor's
counts. Other regional prior rows can remain useful context with model readings
unknown. Derive a tile-specific fraction from the aggregate only when the actual
pinned monitor contains that tile alone, and identify that derivation.

For the fresh CIV89-tile/ten-group contract, original positive polygons are
cashew.kmz (16 polygons), approved prior is CIVv3, fresh start is canonicalE173,
and the cap is200 with early stopping. Source weights stay unchanged. These
facts do not override a historical packet's recorded contract.

The user's regional observation matters when supplied: dense Cashew is mostly
in the strongest prior regions on the common whole-WestAfrica scale. Whole-map
p90/p95/p99 are approximately .844/.867/.917; CIV upper-tail references .92/.94
are a different scope. These are descriptive raw-score references, not calibrated
probabilities, masks or prevalence quotas. Global rarity can coexist with dense
local orchards. Missing prior is unknown. Distinguish canopy pixels, plantation
footprint and the12 mapped countries from all of geographic WestAfrica.

Mixed woody vegetation and wetlands can be confusion contexts. There is no
wetland model output class. External tree cover can include plantations; a map
or agronomic expectation alone does not label a pixel negative. Explain how
model, prior, original source Cashew features and supported human errors agree
or conflict before selecting a useful intervention.

When `woody_wetland_confusion_expert_v1` is supplied, read the
[full approved project guidance](woody-wetland-confusion.md). This versioned
reference carries the domain knowledge requested by the prior helper; the
current packet remains the execution contract.

Automatic negatives are an authorized, separate, low-weight hypothesis only
when the packet advertises that route. Use original source-Cashew AEF distance,
rising p2 and regional prior as soft evidence. Do not add a mandatory high-p2
gate, non-Cashew prototype match, geographic quota or patch-count limit. Preserve
human/evaluation locations and count actual exposure. Human polygons, automatic
pseudo-negatives and historical weak negatives have different reliability and
loss routes. More exposure cannot manufacture missing negative diversity.

## Execution boundary

For research packets advertising an evidence registry, the profile's
[registry adapter](evidence_registry.py) keeps source-validation Cashew F1,
held-out target-panel Cashew F1 and unlabelled monitor responses separate.
Claim checking validates identity and recorded direction; it neither ranks
target quality from unlabelled responses nor proves old state comparability.

Use advertised controls, schema and capabilities. Existing formal scalar trials
discard branch weights and apply selected controls to the main state. A future
state-preparation experiment must explicitly advertise its adoption semantics;
this candidate Skill alone does not authorize warm-state adoption or new losses.
Current scalar-only packets still require a legal complete initial choice.

In the current FADA semantic executor, STOP leaves the formal configuration and
adversarial state unchanged, even for an emergency source-collapse exit. Copy
`current_config.adversarial_state` into `adversarial_state_after`; keep parameter
updates, schedule choices and schedule updates empty. STOP halts updates; it is
not PAUSE_ADV, recovery execution or checkpoint selection.

PAUSE_ADV and RESUME_ADV change the advertised adversarial state without scalar
or schedule updates. A schedule-phase hold/resume operation requires ADJUST;
its presence in the schema does not permit attaching it to PAUSE_ADV. Do not add
a hold merely because adversarial training pauses. Inspect the actual schedule
contract for a later separate action if a phase change is useful. Under a pause,
discriminator/adversarial responses are unmeasured unless a supplied diagnostic
actually measures them; disabled work is not a successful zero-pressure result.

Source collapse floors and STOP rules are packet-versioned. If early sustained
source-degradation guidance conflicts with a catastrophic-only packet, describe
the conflict and follow the actual permitted action set; do not invent authority.
Protecting source semantics is not ranking healthy target checkpoints by source F1.

Use the current stage contract for execution and budgets. Current user-authorized
local research permits physical GPUs0–3 and6, preferring6, at most4 tasks/GPU
and24 machine-wide; the worker checks shared leases and actual processes.
The first behavior-only stage did not authorize GPU training.124 retains its
separate user maximum6/card; existing tasks are not part of this research. No hardware reset, server switch or
unrelated-process termination. Numeric/CPU workers clear all six upper/lowercase
HTTP/HTTPS/ALL proxy variables. Scientific calls use the user's codex_jhy account
and its proxy, never jhyapi. The current knowledge-construction Goal authorizes no new training or scientific model calls. Earlier efficacy-stage plans are historical rather than this phase's completion gate. This candidate is not installed; knowledge construction does not itself replace a frozen production Skill or resume purpose-held work.


## Current research objective and executable selection

The user prioritizes actual recognition quality; cost savings are not required
for a quality improvement. Follow the stage's prospectively declared endpoint;
report FP/FN and costs separately, without rewriting completed endpoints.

For a single-action SELECT_CHECKPOINT turn, the owning caller should use
`packet_contract.specialize_output_schema(request, schema)` before dispatch,
then supply its derived contract to both arms. It binds the integer epoch to
the supplied eligible artifacts. No measured quality winner is required to make
a provisional selection: compare a concrete eligible candidate and alternative,
state the uncertain criterion, and return the selected artifact. Missing target
quality is not permission to return null when this executor requires a selection.
No-eligible-candidate states require a separate advertised failure contract;
do not invent a checkpoint or choose epoch0 outside the permitted set.
