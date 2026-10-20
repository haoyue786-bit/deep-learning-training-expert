"""Small optional mechanisms on existing trainable features and FADA losses.

These are explicitly named project mechanisms, not implementations of entire
MCC/CDAN/CCA-LSC methods. Validation labels are never accepted by loss routes.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def _finite_scalar(value, name, lower=0., upper=None):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"Finite scalar required: {name}")
    if value < lower or (upper is not None and value > upper):
        raise ValueError(f"Out-of-range control: {name}")
    return float(value)


def combine_adversarial(local, global_loss, *, local_scale=1., global_scale=1.):
    local_scale = _finite_scalar(local_scale, "local_scale")
    global_scale = _finite_scalar(global_scale, "global_scale")
    return local_scale*local + global_scale*global_loss


def class_pixel_priority(knowledge, multipliers, valid=None):
    """M02: pixel weights; preserve clipped class proportions, mean priority one."""
    if knowledge.ndim != 4 or len(multipliers) != knowledge.shape[1]:
        raise ValueError("BxCxHxW class knowledge and one priority per class required")
    if not torch.isfinite(knowledge).all() or (knowledge < 0).any():
        raise ValueError("Nonnegative finite class knowledge required")
    values = [_finite_scalar(v, "class_multiplier") for v in multipliers]
    if not any(values): raise ValueError("At least one class must retain alignment support")
    q = knowledge.detach(); mass = q.sum(1)
    mask = mass > 0
    if valid is not None: mask = mask & valid.bool()
    weights = q.new_tensor(values)[None, :, None, None]
    raw = (q*weights).sum(1)/mass.clamp_min(torch.finfo(q.dtype).eps)
    mean = raw[mask].mean() if mask.any() else q.new_tensor(0.)
    if mask.any() and mean <= 0: raise ValueError("Configured priorities remove all supported classes")
    result = torch.where(mask, raw/mean.clamp_min(torch.finfo(q.dtype).eps), torch.zeros_like(raw))
    return result.detach(), dict(supported_pixels=int(mask.sum()),
                                  mean_priority=float(result[mask].mean()) if mask.any() else None,
                                  within_pixel_class_proportions_changed=False)


def pair_ambiguity_reliability(knowledge, *, class_pair, strength=0.5,
                               ecological_support=None, ecological_confidence=None,
                               ecological_strength=0.):
    """M03: detached soft attenuation; no forced normalization, pseudo-label or mask.

    Ecological support is task-supplied relative context, not a crop probability.
    Missing context remains unknown and does not attenuate any pixel.
    """
    strength = _finite_scalar(strength, "strength", upper=1.)
    ecological_strength = _finite_scalar(ecological_strength, "ecological_strength", upper=1.)
    q = knowledge.detach()
    if q.ndim != 4 or not torch.isfinite(q).all() or (q < 0).any():
        raise ValueError("Finite nonnegative BxCxHxW class knowledge required")
    a, b = class_pair
    if a == b or not all(0 <= c < q.shape[1] for c in (a, b)):
        raise ValueError("Two distinct known classes required")
    eps = torch.finfo(q.dtype).eps; mass = q.sum(1); pair_mass = q[:, a]+q[:, b]
    ambiguity = 2*torch.minimum(q[:, a], q[:, b])/pair_mass.clamp_min(eps)
    reliability = 1-strength*(pair_mass/mass.clamp_min(eps))*ambiguity
    context_pixels = 0
    if ecological_support is not None or ecological_confidence is not None:
        if ecological_support is None or ecological_confidence is None:
            raise ValueError("Ecological support requires its confidence/unknown identity")
        support, confidence = torch.broadcast_tensors(torch.as_tensor(ecological_support, device=q.device, dtype=q.dtype),
                                                     torch.as_tensor(ecological_confidence, device=q.device, dtype=q.dtype))
        support = torch.broadcast_to(support, mass.shape); confidence = torch.broadcast_to(confidence, mass.shape)
        known = torch.isfinite(support) & torch.isfinite(confidence)
        if ((confidence[known] < 0) | (confidence[known] > 1) | (support[known] < 0) | (support[known] > 1)).any():
            raise ValueError("Declared relative ecological context/confidence must lie in [0,1]")
        attenuation = ecological_strength*confidence.nan_to_num(0.)*(1-support.nan_to_num(1.))*(q[:, a]/mass.clamp_min(eps))
        reliability = reliability*(1-torch.where(known, attenuation, torch.zeros_like(attenuation)))
        context_pixels = int(known.sum())
    reliability = torch.where(mass > 0, reliability.clamp(0., 1.), torch.zeros_like(reliability)).detach()
    return reliability, dict(mean_reliability=float(reliability.mean()),
                             effective_pixel_mass=float(reliability.sum()), ecological_supported_pixels=context_pixels,
                             normalized_back_to_one=False, generated_target_labels=False)


def conditional_domain_loss(output, knowledge, *, domain, priority=None, reliability=None):
    """Existing FADA joint 2K soft target, with optional declared pixel weights."""
    if output.ndim != 4 or knowledge.ndim != 4 or output.shape[1] != 2*knowledge.shape[1]:
        raise ValueError("FADA requires joint 2K discriminator logits")
    if output.shape[0] != knowledge.shape[0] or output.shape[-2:] != knowledge.shape[-2:]:
        raise ValueError("Align class knowledge with actual feature support before loss")
    if domain not in ("source", "target"): raise ValueError("Explicit discriminator target domain required")
    q = knowledge.detach(); zero = torch.zeros_like(q)
    encoded = torch.cat((q, zero) if domain == "source" else (zero, q), dim=1)
    raw = -(encoded*F.log_softmax(output, dim=1)).sum(1)
    valid = q.sum(1) > 0; weights = torch.ones_like(raw)
    for value in (priority, reliability):
        if value is not None:
            if value.shape != raw.shape or not torch.isfinite(value).all() or (value < 0).any():
                raise ValueError("Finite nonnegative pixel weights matching discriminator output required")
            weights = weights*value.detach()
    # Divide by valid pixel count, never by reliability mass: attenuation stays real.
    loss = (raw*weights)[valid].mean() if valid.any() else output.sum()*0.
    return loss, dict(supported=bool(valid.any()), supported_pixels=int(valid.sum()),
                     effective_weight_mass=float(weights[valid].sum()),
                     denominator="valid pixels; reliability not renormalized")


def source_pair_margin(features, labels, *, class_pair, margin=0.3, data_role="source_training"):
    """M04: trainable source features, detached batch prototypes and cosine margin.

    No target pseudo-labels or validation labels are used. Missing pair support
    is explicitly unsupported; it is not a successful zero-loss observation.
    """
    if data_role != "source_training": raise PermissionError("This kernel permits source training truth only")
    margin = _finite_scalar(margin, "margin", upper=2.)
    if features.ndim != 4 or labels.ndim != 3 or labels.shape[0] != features.shape[0]:
        raise ValueError("Differentiable feature map and spatial source labels required")
    labels = F.interpolate(labels[:, None].float(), size=features.shape[-2:], mode="nearest")[:, 0].long()
    a, b = class_pair
    if a == b or min(a, b) < 0: raise ValueError("Distinct source class pair required")
    pixels = features.permute(0, 2, 3, 1); ma = labels == a; mb = labels == b
    counts = {str(a): int(ma.sum()), str(b): int(mb.sum())}
    if not ma.any() or not mb.any():
        return features.sum()*0., dict(supported=False, reason="source class pair absent", class_support=counts)
    pa = F.normalize(pixels[ma].detach().mean(0), dim=0); pb = F.normalize(pixels[mb].detach().mean(0), dim=0)
    va = F.normalize(pixels[ma], dim=1); vb = F.normalize(pixels[mb], dim=1)
    loss = 0.5*(F.relu(margin+(va@pb)-(va@pa)).mean() + F.relu(margin+(vb@pa)-(vb@pb)).mean())
    return loss, dict(supported=True, class_support=counts, prototype_cosine=float(pa@pb),
                     source_feature_gradient_path=bool(features.requires_grad),
                     labels_from_validation=False, prototype_policy="detached current-source-batch means")


def source_hard_negative_loss(logits, labels, *, focus_class, negative_class, fraction=0.25,
                              data_role="source_training", valid=None):
    """M08 source-only hard negatives; labels never come from target validation."""
    if data_role not in ("source_training", "human_training"):
        raise PermissionError("Hard-negative labels must be authorized training annotations")
    fraction = _finite_scalar(fraction, "fraction", upper=1.)
    if fraction == 0: raise ValueError("Use disable for zero exposure")
    if logits.shape[0] != labels.shape[0] or logits.shape[-2:] != labels.shape[-2:]:
        raise ValueError("Training masks must match segmentation output")
    if focus_class == negative_class or not all(0 <= c < logits.shape[1] for c in (focus_class, negative_class)):
        raise ValueError("Distinct declared positive/negative classes required")
    mask = labels == negative_class
    if valid is not None: mask = mask & valid.bool()
    log_p = F.log_softmax(logits, 1)[:, focus_class]
    # -log(1-p) with stable complementary logsumexp; no forced negative subclass.
    others = [c for c in range(logits.shape[1]) if c != focus_class]
    raw = torch.logsumexp(logits, 1)-torch.logsumexp(logits[:, others], 1)
    count = int(mask.sum())
    if not count: return logits.sum()*0., dict(supported=False, eligible_pixels=0, selected_pixels=0)
    take = max(1, int(math.ceil(fraction*count)))
    indices = torch.topk(log_p.detach()[mask], take, sorted=False).indices
    return raw[mask][indices].mean(), dict(supported=True, eligible_pixels=count, selected_pixels=take,
                                         origin=data_role, generated_target_labels=False)
