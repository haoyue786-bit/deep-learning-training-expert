"""Verify immutable file bindings before reusing a completed decision.

This is an integrity check, not a decision validator or a retry controller.
It reads only the three supplied files and never invokes a model or training.
"""
import argparse
import hashlib
import json
from pathlib import Path


class ReceiptError(ValueError):
    pass


def _read(path):
    try:
        data = Path(path).read_bytes()
        value = json.loads(data)
    except (OSError, ValueError) as exc:
        raise ReceiptError("Missing or invalid receipt artifact: %s" % path) from exc
    if not isinstance(value, dict):
        raise ReceiptError("Receipt artifact must be a JSON object: %s" % path)
    return data, value


def verify_completion(request_path, decision_path, provenance_path, expected_context=None):
    paths = [Path(x) for x in (request_path, decision_path, provenance_path)]
    if len({x.resolve() for x in paths}) != 3:
        raise ReceiptError("Request, decision and provenance must be distinct artifacts")
    captured = [_read(x) for x in paths]
    request_bytes, request = captured[0]
    decision_bytes, decision = captured[1]
    _, provenance = captured[2]
    request_hash = hashlib.sha256(request_bytes).hexdigest()
    decision_hash = hashlib.sha256(decision_bytes).hexdigest()
    if decision.get("request_sha256") != request_hash:
        raise ReceiptError("Decision does not bind the supplied request bytes")
    if provenance.get("request_sha256") != request_hash:
        raise ReceiptError("Provenance does not bind the supplied request bytes")
    if provenance.get("decision_sha256") != decision_hash:
        raise ReceiptError("Provenance does not bind the supplied decision bytes")
    for key, value in (expected_context or {}).items():
        if provenance.get(key) != value:
            raise ReceiptError("Completion context mismatch: %s" % key)
    # Detect replacement while the three-file receipt was being read. No claim
    # is made about hostile concurrent writers or a later mutation by callers.
    for path, (data, _) in zip(paths, captured):
        if path.read_bytes() != data:
            raise ReceiptError("Receipt changed during verification: %s" % path)
    return {"state": "verified_binding", "request_sha256": request_hash,
            "decision_sha256": decision_hash, "decision": decision,
            "provenance": provenance,
            "limits": "Byte/context binding only; scientific validity and permission require separate checks"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--decision", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    args = parser.parse_args()
    result = verify_completion(args.request, args.decision, args.provenance)
    print(json.dumps({k: v for k, v in result.items() if k not in {"decision", "provenance"}}, indent=2))


if __name__ == "__main__":
    main()
