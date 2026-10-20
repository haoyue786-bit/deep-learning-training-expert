"""Project-authorized, append-only scientific experience and validation usage.

No project, filesystem location, data permission or execution authority is inferred.
Run data never belong in the installed Skill directory.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def file_evidence(path: str | Path) -> dict:
    path = Path(path).resolve(strict=True)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return {"path": str(path), "sha256": h.hexdigest(), "bytes": path.stat().st_size}


class ProjectStore:
    """Resolve an explicit authorization and reject missing/mixed project identity."""

    def __init__(self, context: dict):
        required = {"project_id", "project_root", "experience_dir", "authorization"}
        if not required <= context.keys():
            raise ValueError("Explicit project identity, root, experience_dir and authorization required")
        self.project_id = context["project_id"]
        if not isinstance(self.project_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", self.project_id):
            raise ValueError("Invalid project_id")
        auth = context["authorization"]
        if not isinstance(auth, dict) or auth.get("write_experience") is not True or not auth.get("source"):
            raise PermissionError("Project-specific experience write authorization is missing")
        for key in ("project_root", "experience_dir"):
            if not isinstance(context[key], str) or not Path(context[key]).is_absolute():
                raise ValueError(f"{key} must be an explicit absolute path")
        self.project_root = Path(context["project_root"]).resolve(strict=True)
        self.root = Path(context["experience_dir"]).resolve()
        if self.root == self.project_root or not within(self.root, self.project_root):
            raise PermissionError("experience_dir must be a dedicated directory inside the authorized project")
        # Skill/repository code is never an implicit destination for run outcomes.
        if (any(part in (".codex", ".codex_jhy", ".codex_jhyapi") for part in self.root.parts)
                or within(self.root, Path(__file__).resolve().parents[2])):
            raise PermissionError("Experience cannot be written into a Codex installation")
        self.context = dict(context)

    @classmethod
    def from_file(cls, path: str | Path) -> "ProjectStore":
        return cls(json.loads(Path(path).read_text()))

    @contextmanager
    def locked(self):
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.resolve() != self.root:
            raise PermissionError("Experience directory resolution changed")
        with (self.root / ".project.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            identity = self.root / "project_identity.json"
            expected = {"project_id": self.project_id, "project_root": str(self.project_root),
                        "experience_dir": str(self.root)}
            if identity.exists():
                if json.loads(identity.read_text()) != expected:
                    raise PermissionError("Existing store belongs to a different project")
            else:
                self._atomic(identity, expected)
            yield

    @staticmethod
    def _atomic(path: Path, value: dict):
        temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)

    def _read(self, name: str) -> list[dict]:
        path = self.root / name
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def _append(self, name: str, value: dict):
        with (self.root / name).open("a") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def record(self, case: dict) -> dict:
        required = {"case_id", "project_id", "action", "result", "outcome", "evidence",
                    "conditions", "hypotheses", "cost", "evaluation_role"}
        if not required <= case.keys():
            raise ValueError(f"Missing experience fields: {sorted(required-case.keys())}")
        if case["project_id"] != self.project_id:
            raise PermissionError("Case project identity does not match authorized store")
        if case["outcome"] not in ("positive", "null", "adverse", "failed", "incomplete", "observation"):
            raise ValueError("Explicit positive/null/adverse/failed/incomplete/observation outcome required")
        if case["evaluation_role"] not in ("development", "independent_evaluation", "training_diagnostic", "no_quality_evaluation"):
            raise ValueError("Explicit evaluation role required")
        if not isinstance(case["conditions"], dict) or not isinstance(case["hypotheses"], list):
            raise ValueError("Conditions and unverified hypotheses must remain separate")
        if not case["evidence"]:
            raise ValueError("At least one actual evidence artifact required, including for failures")
        for item in case["evidence"]:
            actual = file_evidence(item["path"])
            if actual["sha256"] != item["sha256"]:
                raise ValueError("Experience evidence changed")
        body_hash = digest(case)
        with self.locked():
            for old in self._read("experience.jsonl"):
                if old["case_id"] == case["case_id"]:
                    if old["body_sha256"] != body_hash:
                        raise ValueError("Cannot rewrite an existing case; append a correction with a new case_id")
                    return old
            result = dict(case, body_sha256=body_hash, recorded_at=datetime.now(timezone.utc).isoformat())
            self._append("experience.jsonl", result)
            return result

    def reserve_validation(self, *, usage_id: str, dataset: dict, checkpoint: dict,
                           purpose: str, maximum_uses: int) -> dict:
        """Reserve before label feedback; failures still count. Exact requests are idempotent."""
        if purpose not in ("baseline_diagnosis", "module_screening", "trial_evaluation", "online_review"):
            raise PermissionError("This development ledger does not authorize threshold searches or final TEST use")
        if dataset.get("role") != "development" or dataset.get("labels_for_gradient", False):
            raise PermissionError("Only declared development feedback, never label backpropagation")
        if not dataset.get("dataset_id") or not dataset.get("label_version") or not dataset.get("sha256"):
            raise ValueError("Dataset identity and label version/hash required")
        if not checkpoint.get("identity") or not checkpoint.get("sha256"):
            raise ValueError("Checkpoint identity/hash required")
        if isinstance(maximum_uses, bool) or not isinstance(maximum_uses, int) or maximum_uses <= 0:
            raise ValueError("A finite positive validation-use budget required")
        body = dict(project_id=self.project_id, usage_id=usage_id, dataset=dataset,
                    checkpoint=checkpoint, purpose=purpose, maximum_uses=maximum_uses)
        body_hash = digest(body)
        with self.locked():
            events = self._read("validation_usage.jsonl")
            for event in events:
                if event.get("event") == "reserved" and event["usage_id"] == usage_id:
                    if event["request_sha256"] != body_hash:
                        raise ValueError("Validation usage_id reused for different evidence")
                    return event
            reservations = [e for e in events if e.get("event") == "reserved"
                            and e["dataset"]["dataset_id"] == dataset["dataset_id"]]
            # Relabeling or increasing the caller budget does not reset prior exposure.
            limit = min([maximum_uses] + [e["maximum_uses"] for e in reservations])
            if len(reservations) >= limit:
                raise PermissionError("Validation development-use budget exhausted; no automatic search expansion")
            result = dict(body, event="reserved", ordinal=len(reservations)+1,
                          request_sha256=body_hash, recorded_at=datetime.now(timezone.utc).isoformat())
            self._append("validation_usage.jsonl", result)
            return result

    def finish_validation(self, usage_id: str, artifact: str | Path, *, status: str = "completed") -> dict:
        if status not in ("completed", "failed", "incomplete"):
            raise ValueError("Unknown feedback outcome")
        evidence = file_evidence(artifact)
        with self.locked():
            events = self._read("validation_usage.jsonl")
            if not any(e.get("event") == "reserved" and e["usage_id"] == usage_id for e in events):
                raise ValueError("Reserve validation feedback before recording its result")
            result = dict(project_id=self.project_id, usage_id=usage_id, event="result", status=status,
                          evidence=evidence, recorded_at=datetime.now(timezone.utc).isoformat())
            previous = [e for e in events if e.get("event") == "result" and e["usage_id"] == usage_id]
            if previous:
                if previous[-1]["evidence"] != evidence or previous[-1]["status"] != status:
                    raise ValueError("Cannot rewrite a feedback result")
                return previous[-1]
            self._append("validation_usage.jsonl", result)
            return result

    def retrieve(self, *, tags: list[str] = (), limit: int = 5) -> list[dict]:
        """Return only this project's evidence; matching tags are clues, not causal transfer."""
        if not isinstance(limit, int) or not 0 < limit <= 100:
            raise ValueError("Bounded retrieval limit required")
        with self.locked():
            cases = self._read("experience.jsonl")
        if any(c.get("project_id") != self.project_id for c in cases):
            raise PermissionError("Project store contains foreign-project records")
        return sorted(cases, key=lambda c: (len(set(tags)&set(c.get("tags", []))), c["recorded_at"]), reverse=True)[:limit]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-context", required=True, type=Path)
    sub = parser.add_subparsers(dest="action", required=True)
    record = sub.add_parser("record"); record.add_argument("--case", required=True, type=Path)
    retrieve = sub.add_parser("retrieve"); retrieve.add_argument("--tag", action="append", default=[])
    retrieve.add_argument("--limit", type=int, default=5)
    args = parser.parse_args(); store = ProjectStore.from_file(args.project_context)
    result = store.record(json.loads(args.case.read_text())) if args.action == "record" else store.retrieve(tags=args.tag, limit=args.limit)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
