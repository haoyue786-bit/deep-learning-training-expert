"""One capability registry for cards, controls, validation and callable identity."""
from __future__ import annotations

from dataclasses import dataclass
import copy
import math


@dataclass(frozen=True)
class Capability:
    module_id: str
    variant: str
    implementation: str | None
    controls: dict
    input_roles: tuple
    status: str
    mechanism: str
    reversible: str = "Disabling stops future loss; undoing updates requires full-state restore."


CAPABILITIES = (
    Capability("M01", "adversarial_path_balance", "combine_adversarial",
               {"local_scale": (0., 4., 1.), "global_scale": (0., 4., 1.)}, ("unlabelled_training",),
               "bench_kernel", "Separate model-side local/global loss scales; discriminator updates remain separate."),
    Capability("M02", "class_pixel_priority", "class_pixel_priority", {}, ("unlabelled_training",),
               "bench_kernel", "Detached mean-one pixel priority preserves clipped within-pixel class proportions."),
    Capability("M03", "pair_ambiguity_reliability", "pair_ambiguity_reliability",
               {"strength": (0., 1., .5), "ecological_strength": (0., 1., 0.)}, ("unlabelled_training",),
               "bench_kernel", "Soft attenuation of ambiguous class conditioning; ecology remains optional uncertain context."),
    Capability("M04", "source_pair_margin", "source_pair_margin", {"margin": (0., 2., .3)}, ("source_training",),
               "bench_kernel", "Source-supervised cosine-margin separation on actual trainable features."),
    Capability("M05", "class_mass_correction", None, {}, (), "design_only",
               "Needs supported class-mass assumptions; predicted means are not target truth."),
    Capability("M06", "source_gradient_projection", None, {}, ("source_training", "unlabelled_training"),
               "existing_external_adapter_required", "Reuse the pinned gradient_control_v1 executor; no duplicate projection implementation."),
    Capability("M07", "component_learning_policy", None, {}, (), "existing_external_adapter_required",
               "Reuse component groups, buffer/mode preservation and full-state restoration from the host."),
    Capability("M08", "source_hard_negative", "source_hard_negative_loss", {"fraction": (.01, 1., .25)},
               ("source_training", "human_training"), "bench_kernel",
               "Mine known training negatives by focus-class error; never assign target validation labels to loss."),
    Capability("M08", "automatic_negative_pool", None, {}, ("unlabelled_training",),
               "existing_external_adapter_required", "Reuse versioned selection/protection; align refresh and finite exposure with Agent cadence."),
)


def get(module_id: str, variant: str) -> Capability:
    found = [c for c in CAPABILITIES if (c.module_id, c.variant) == (module_id, variant)]
    if len(found) != 1: raise ValueError("Unknown or ambiguous module/variant")
    return found[0]


def controls_schema(module_id: str, variant: str) -> dict:
    cap = get(module_id, variant)
    properties={name: dict(type="number", minimum=v[0], maximum=v[1], default=v[2]) for name, v in cap.controls.items()}
    if module_id=='M02':
        properties['multipliers']=dict(type='array',minItems=2,items=dict(type='number',minimum=0,maximum=4),
                                       description='One priority per actual class; omit for neutral all-ones in invoke().')
    return dict(type="object", additionalProperties=False, properties=properties)


def resolve(module_id: str, variant: str, *, controls=None, data_role: str,
            authorized_variants: list[str], execution_scope: str = "trial") -> tuple:
    cap = get(module_id, variant)
    if execution_scope != "trial":
        raise PermissionError("Bench kernels are not registered production interventions")
    if f"{module_id}:{variant}" not in authorized_variants:
        raise PermissionError("Module variant not authorized in the trial contract")
    if cap.implementation is None: raise ValueError("This capability requires an existing version-bound host adapter")
    if data_role not in cap.input_roles: raise PermissionError("Module input has a forbidden data role")
    controls = copy.deepcopy(controls or {})
    multipliers=controls.pop('multipliers',None) if module_id=='M02' else None
    unknown = set(controls)-set(cap.controls)
    if unknown: raise ValueError(f"Unknown controls: {sorted(unknown)}")
    result = {}
    for name, (lower, upper, default) in cap.controls.items():
        value = controls.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"Invalid control: {name}")
        result[name] = float(value)
    if multipliers is not None:
        if not isinstance(multipliers,list) or len(multipliers)<2 or not all(
                isinstance(v,(float,int)) and not isinstance(v,bool) and math.isfinite(v) and 0<=v<=4 for v in multipliers):
            raise ValueError('One finite bounded class multiplier per actual class required')
        result['multipliers']=multipliers
    from . import algorithms
    return getattr(algorithms, cap.implementation), result


def invoke(module_id: str, variant: str, *, inputs: dict, controls=None, data_role: str,
           authorized_variants: list[str], execution_scope='trial'):
    """One callable interface; no model request, extra Agent or dynamic code eval."""
    function,values=resolve(module_id,variant,controls=controls,data_role=data_role,
                            authorized_variants=authorized_variants,execution_scope=execution_scope)
    if module_id=='M02' and 'multipliers' not in values:
        values['multipliers']=[1.]*inputs['knowledge'].shape[1]
    if module_id in ('M04','M08'):values['data_role']=data_role
    return function(**inputs,**values)


def cards(*, available_variants: list[str] = (), scope="trial") -> list[dict]:
    return [dict(module_id=c.module_id, variant=c.variant, status=c.status,
                 callable=scope == "trial" and c.implementation is not None and f"{c.module_id}:{c.variant}" in available_variants,
                 production_integrated=False, mechanism=c.mechanism, input_roles=list(c.input_roles),
                 controls_schema=controls_schema(c.module_id, c.variant), restore_semantics=c.reversible)
            for c in CAPABILITIES]
