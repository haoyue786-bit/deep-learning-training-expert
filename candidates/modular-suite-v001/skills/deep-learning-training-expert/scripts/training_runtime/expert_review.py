"""Compact exact observations and whole-history plots. No training/model calls."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
from pathlib import Path

IDENTITIES = ("run_id", "branch_id", "data_identity")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serialize(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def section(path, heading, maximum_characters=2800):
    path = Path(path).resolve()
    lines = path.read_text().splitlines()
    matches = [i for i, line in enumerate(lines) if line.lstrip("#").strip() == heading and line.startswith("#")]
    if len(matches) != 1:
        raise ValueError("One exact unique heading required")
    start = matches[0]; level = len(lines[start]) - len(lines[start].lstrip("#"))
    end = len(lines)
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if line.startswith("#") and len(line) - len(line.lstrip("#")) <= level:
            end = index; break
    text = "\n".join(lines[start:end]).strip()
    if len(text) > maximum_characters:
        raise ValueError("Select a narrower relevant knowledge section")
    return dict(source=str(path), sha256=sha(path), heading=heading, text=text)


def normalize(data):
    identity = {k: data[k] for k in IDENTITIES}
    if any(not v for v in identity.values()):
        raise ValueError("Explicit run, branch and data/decoder/mask identity required")
    cutoff = data["cutoff_epoch"]
    if type(cutoff) is not int or cutoff < 0:
        raise ValueError("Nonnegative integer evidence cutoff required")
    specs = data["metric_specs"]
    keys = [s["key"] for s in specs]
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("Nonempty unique metric definitions required")
    for spec in specs:
        if any(not isinstance(spec.get(k), str) or not spec[k] for k in ("key", "label", "unit", "panel", "definition")):
            raise ValueError("Metric units, scope and definition required")
        if spec.get("direction") not in ("max", "min", "none"):
            raise ValueError("Declare quality direction or none for diagnostics")
    if len(set(s["panel"] for s in specs)) > 12:
        raise ValueError("Use at most twelve legible panels; move layer details to an anomaly review")
    rows = []; excluded = 0
    for item in data["observations"]:
        if any(item.get(k) != v for k, v in identity.items()):
            raise ValueError("Mixed run/branch/data semantics; create separate trajectories")
        if type(item.get("epoch")) is not int or item["epoch"] < 0 or not item.get("checkpoint"):
            raise ValueError("Every observed epoch requires its actual checkpoint identity")
        if item["epoch"] > cutoff:
            excluded += 1; continue
        if not isinstance(item.get("metrics"), dict) or set(item["metrics"]) - set(keys):
            raise ValueError("Every recorded metric requires an explicit definition")
        for value in item["metrics"].values():
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                raise ValueError("Nonfinite data must become a recorded anomaly, never a hidden plot value")
        rows.append(copy.deepcopy(item))
    rows.sort(key=lambda r: r["epoch"])
    if not rows or len({r["epoch"] for r in rows}) != len(rows) or rows[-1]["epoch"] != cutoff:
        raise ValueError("Unique observed epochs ending at the actual review cutoff required")
    events = []
    for event in data.get("events", []):
        if any(event.get(k) != v for k, v in identity.items()):
            raise ValueError("Foreign intervention history")
        if type(event.get("epoch")) is not int or event["epoch"] < 0:
            raise ValueError("Event timing required")
        if event["epoch"] > cutoff:
            continue
        if event.get("status") not in ("applied", "proposed", "rejected") or not event.get("summary"):
            raise ValueError("Explicit event status and summary required")
        events.append(copy.deepcopy(event))
    return identity, specs, rows, events, excluded


def compact_history(specs, rows, events):
    anchors = {0, len(rows) - 1}
    anchors.update(range(max(0, len(rows) - 5), len(rows)))
    summary = {}; missing = []
    for spec in specs:
        key = spec["key"]
        valid = [(i, row["metrics"].get(key)) for i, row in enumerate(rows) if row["metrics"].get(key) is not None]
        if not valid:
            missing.append(key); continue
        low = min(valid, key=lambda p: p[1]); high = max(valid, key=lambda p: p[1])
        anchors.update((valid[0][0], valid[-1][0], low[0], high[0]))
        recent = valid[-5:]
        record = dict(n=len(valid), first=[rows[valid[0][0]]["epoch"], valid[0][1]],
            latest=[rows[valid[-1][0]]["epoch"], valid[-1][1]],
            min=[rows[low[0]]["epoch"], low[1]], max=[rows[high[0]]["epoch"], high[1]],
            recent_endpoint_change=recent[-1][1]-recent[0][1] if len(recent) > 1 else None,
            missing_epochs=len(rows)-len(valid))
        if spec["direction"] != "none":
            best = high if spec["direction"] == "max" else low
            record["best_observed"] = [rows[best[0]]["epoch"], best[1]]
        summary[key] = record
    for event in events:
        if event["status"] != "applied":
            continue
        before = [i for i, r in enumerate(rows) if r["epoch"] < event["epoch"]]
        after = [i for i, r in enumerate(rows) if r["epoch"] >= event["epoch"]]
        if before: anchors.add(before[-1])
        if after: anchors.add(after[0])
    return summary, missing, [rows[i] for i in sorted(anchors)]


def plot_history(path, specs, rows, events, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    all_panels = list(dict.fromkeys(s["panel"] for s in specs))
    panels = [p for p in all_panels if any(r["metrics"].get(s["key"]) is not None
              for s in specs if s["panel"] == p for r in rows)]
    absent = [p for p in all_panels if p not in panels]
    if absent: panels.append("Not yet measured")
    nrows = math.ceil(len(panels)/2)
    fig, axes = plt.subplots(nrows, 2, figsize=(14, 2.6*nrows), squeeze=False)
    epochs = [r["epoch"] for r in rows]
    applied = [e for e in events if e["status"] == "applied"]
    for ax, panel in zip(axes.flat, panels):
        if panel == "Not yet measured":
            ax.axis("off")
            ax.set_title(panel,loc="left",fontsize=10)
            ax.text(.02,.85,"\n".join(absent)+"\n\nUnknown is not zero.",ha="left",va="top",
                    transform=ax.transAxes,fontsize=9,color="#666")
            continue
        panel_specs = [s for s in specs if s["panel"] == panel]
        plotted = False
        for spec in panel_specs:
            values = [r["metrics"].get(spec["key"]) for r in rows]
            if not any(v is not None for v in values):
                continue
            ax.plot(epochs, [np.nan if v is None else v for v in values], marker=".", markersize=5,
                    linewidth=1.25, label=spec["label"])
            plotted = True
        for event in applied:
            ax.axvline(event["epoch"], color="#9a6700", alpha=.45, linewidth=1, linestyle="--")
        ax.set_title(panel, loc="left", fontsize=10)
        units = sorted(set(s["unit"] for s in panel_specs))
        ax.set_ylabel(" / ".join(units), fontsize=8)
        ax.set_xlabel("Completed epoch", fontsize=8)
        ax.grid(alpha=.18)
        if plotted: ax.legend(fontsize=7, loc="best")
        else: ax.text(.5, .5, "Not measured", ha="center", va="center", transform=ax.transAxes, color="#777")
        if len(rows) == 1:
            ax.set_xlim(max(-.5, epochs[0]-.5), epochs[0]+.5)
    for ax in list(axes.flat)[len(panels):]: ax.set_visible(False)
    subtitle = "One observation: no training trend yet" if len(rows) == 1 else "All observed epochs; gaps remain missing; no smoothing"
    fig.suptitle(title + "\n" + subtitle, fontsize=12)
    fig.tight_layout(rect=(0, .025, 1, .955))
    fig.text(.02, .01, "Dashed markers = applied interventions only. Exact values/definitions accompany this plot.", fontsize=8)
    fig.savefig(path, dpi=125)
    plt.close(fig)


def build(data, output, maximum_characters=16000):
    identity, specs, rows, events, excluded = normalize(data)
    summary, missing, anchors = compact_history(specs, rows, events)
    knowledge = data.get("knowledge", [])
    experience = data.get("experience", [])
    if len(knowledge) > 2 or len(experience) > 2:
        raise ValueError("Only relevant sections/lessons; no complete manual packet")
    for item in knowledge:
        if any(not item.get(k) for k in ("source", "sha256", "heading", "text")):
            raise ValueError("Knowledge needs delivered text and exact section/source identity")
        if sha(item["source"]) != item["sha256"]:
            raise ValueError("Knowledge source changed")
        exact = section(item["source"], item["heading"], maximum_characters=2800)
        if exact != item:
            raise ValueError("Knowledge excerpt differs from its declared exact section")
    if len(serialize(knowledge)) > 6500 or len(serialize(experience)) > 1800:
        raise ValueError("Knowledge/experience exceeds the compact review budget")
    root = Path(output).resolve()
    if root.exists():
        raise ValueError("Use a new immutable review directory")
    history = dict(**identity, metric_specs=specs, observations=rows, events=events)
    history_text = serialize(history)
    history_hash = hashlib.sha256(history_text.encode()).hexdigest()
    brief = dict(format="expert_review_v1", **identity, epoch=data["cutoff_epoch"],
        task=data["task"], controls=data["controls"], bounds=data["bounds"],
        current=rows[-1], metric_definitions=specs, trajectory_summary=summary,
        exact_anchors=[dict(epoch=r["epoch"],metrics=r["metrics"],checkpoint=r["checkpoint"]) for r in anchors],
        missing_measurements=missing, events=[{k:e[k] for k in ("epoch","status","summary")} for e in events],
        prior_context=data.get("prior_context", {"status":"unknown"}),
        open_questions=data.get("open_questions", []), knowledge=knowledge, experience=experience,
        history=dict(path=str(root/"history.json"),sha256=history_hash,observations=len(rows),
            excluded_future_rows=excluded,stage="E0 only; no trend" if len(rows)==1 else "observed trajectory",
            summary_is_not_causal_inference=True),
        image=dict(path=str(root/"training_curves.png"),sha256="pending",delivery="not_yet_attached"))
    # Reserve for the real image digest and prompt delivery metadata.
    if len(serialize(brief)) + 250 > maximum_characters:
        raise ValueError("Review exceeds budget; explicitly narrow the live question, never silently truncate history")
    root.mkdir(parents=True)
    (root/"history.json").write_text(history_text)
    with (root/"history.csv").open("w", newline="") as stream:
        fields=["epoch","checkpoint"]+[s["key"] for s in specs]
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:writer.writerow(dict(epoch=row["epoch"],checkpoint=row["checkpoint"],**row["metrics"]))
    plot_history(root/"training_curves.png",specs,rows,events,identity["run_id"]+" | "+identity["branch_id"])
    brief["image"]["sha256"]=sha(root/"training_curves.png")
    brief["size"]=dict(characters=len(serialize(brief)),actual_tokens=None)
    (root/"brief.json").write_text(serialize(brief))
    return brief


def model_inputs(brief_path, *, vision_available):
    if type(vision_available) is not bool:
        raise ValueError("Caller must declare actual transport vision support")
    brief=json.loads(Path(brief_path).read_text())
    if sha(brief["history"]["path"]) != brief["history"]["sha256"]:
        raise ValueError("History identity changed")
    image=brief["image"]
    if sha(image["path"]) != image["sha256"]:
        raise ValueError("Curve artifact changed")
    brief["image"]["delivery"]="attached_in_this_request" if vision_available else "not_delivered_text_only"
    inputs=[dict(type="text",text=serialize(brief))]
    if vision_available:inputs.append(dict(type="localImage",path=image["path"]))
    return inputs
