"""Fail-closed validation for Evidence-to-Claim review envelopes."""

from __future__ import annotations

from typing import Any, Iterable


ALLOWED_CLASSIFICATIONS = frozenset({
    "contradiction", "unsupported", "supported", "insufficient_evidence",
})


def validate_review(review: dict[str, Any], finding_ids: Iterable[str],
                    control_ids: Iterable[str]) -> str | None:
    """Return a schema error, or None when the review envelope is valid."""
    finding_ids = set(finding_ids)
    control_ids = set(control_ids)
    if set(review) != {"findings", "negative_controls", "insufficiency_control"}:
        return "review root keys do not match the schema"
    findings = review.get("findings")
    if not isinstance(findings, list) or len(findings) != len(finding_ids):
        return "findings count does not match the schema"
    if any(not isinstance(item, dict) for item in findings):
        return "findings contain a non-object item"
    if {item.get("id") for item in findings} != finding_ids:
        return "findings IDs do not match the schema"
    required_finding_keys = {"id", "detected", "classification", "evidence_ids",
                             "uncertainty", "correction", "abstain"}
    for item in findings:
        if set(item) != required_finding_keys:
            return f"finding {item.get('id')} keys do not match the schema"
        if not isinstance(item["detected"], bool) or item["classification"] not in ALLOWED_CLASSIFICATIONS:
            return f"finding {item.get('id')} has invalid classification or detected type"
        if (not isinstance(item["evidence_ids"], list)
                or not all(isinstance(value, str) for value in item["evidence_ids"])):
            return f"finding {item.get('id')} evidence_ids must be an array of strings"
        if not isinstance(item["uncertainty"], str) or not item["uncertainty"].strip():
            return f"finding {item.get('id')} uncertainty must be non-empty"
        if item["correction"] is not None and not isinstance(item["correction"], str):
            return f"finding {item.get('id')} correction must be a string or null"
        if not isinstance(item["abstain"], bool):
            return f"finding {item.get('id')} abstain must be boolean"
    controls = review.get("negative_controls")
    if (not isinstance(controls, list) or len(controls) != len(control_ids)
            or any(not isinstance(item, dict) for item in controls)
            or {item.get("id") for item in controls} != control_ids):
        return "negative_controls IDs do not match the schema"
    for item in controls:
        if set(item) != {"id", "classification", "evidence_ids"}:
            return f"control {item.get('id')} keys do not match the schema"
        if item["classification"] not in ALLOWED_CLASSIFICATIONS or not isinstance(item["evidence_ids"], list):
            return f"control {item.get('id')} has invalid fields"
    insufficiency = review.get("insufficiency_control")
    if not isinstance(insufficiency, dict) or set(insufficiency) != {
        "id", "classification", "abstain", "evidence_ids", "reason"
    }:
        return "insufficiency_control keys do not match the schema"
    if (insufficiency["id"] != "I1"
            or insufficiency["classification"] != "insufficient_evidence"
            or insufficiency["abstain"] is not True
            or not isinstance(insufficiency["evidence_ids"], list)
            or not isinstance(insufficiency["reason"], str)
            or not insufficiency["reason"].strip()):
        return "I1 must be insufficient_evidence with abstain=true and a reason"
    return None
