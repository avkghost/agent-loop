#!/usr/bin/env python3
"""Evidence-to-Claim pilot for the second, Axum follow-up fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from tools.run_evidence_claim_review import (
    ReviewTransportError,
    _load_session,
    _openai_sdk,
    _response_metadata,
    _response_payload,
    _safe_endpoint,
    _tool_content,
    normalize_endpoint,
)

SESSION_ID = "29d7d1fd-b2ac-4d53-b061-7f198aac076a"
RUN_ID = "run_e7d0615eaa0d49babf710d7b68d65afd"
DEFAULT_SESSION = Path.home() / ".agent_loop" / "sessions" / f"{SESSION_ID}.json"
EXPECTED_FINAL_SHA256 = "3a8b4bcf8b6d0389f7161edae0a3ef20c0581f25381ff5fc4be468e622b67bc0"
EXPECTED_BUNDLE_SHA256 = "3fde4388c839a6de51d953d57ae30a853054199a01829f2be1ee24c4179186da"

REVIEW_REQUEST = """Review the target answer against the evidence bundle.

Return exactly one JSON object and nothing else: no Markdown fences, no prose,
no comments, and no trailing commas. Do not invent keys or classification values.
Return exactly this JSON shape:
{
  "findings": [{"id":"F1|F2|F3", "detected":true,
    "classification":"contradiction|unsupported|supported|insufficient_evidence",
    "evidence_ids":["E1"], "uncertainty":"non-empty phrase",
    "correction":"...", "abstain":false}],
  "negative_controls": [{"id":"N1", "classification":"supported",
    "evidence_ids":["E1"]}],
  "insufficiency_control": {"id":"I1",
    "classification":"insufficient_evidence", "abstain":true,
    "evidence_ids":[], "reason":"non-empty phrase"}
}

Use only these exact classification strings: `contradiction`, `unsupported`,
`supported`, or `insufficient_evidence`. Each finding must include id,
detected, classification, evidence_ids, uncertainty, correction, and abstain.
Before returning, verify that F1, F2, and F3 each have the literal
`uncertainty`: `direct evidence`, and that I1 has the literal
`classification`: `insufficient_evidence`, `abstain`: true, and a non-empty
`reason`. The `reason` value must be a quoted JSON string, for example
`"reason":"The bundle does not establish a project-specific patch."`; never
emit bare prose after the colon. Do not substitute `unsupported` for I1: its
claim is intentionally about evidence insufficiency, not contradiction.

F1: "axum::Server remains available in Axum 0.7."
F2: "Axum 0.7 still re-exports hyper::Body."
F3: "Routers nested with nest_service inherit the outer fallback."
N1: "Axum 0.7 uses axum::serve(listener, service)."
I1: "The evidence establishes the complete project-specific migration patch."

Expected boundaries for this fixture:
- F1 is contradiction: E1 says axum::Server was removed.
- F2 is contradiction: E2 says axum no longer re-exports hyper::Body.
- F3 is contradiction: E3 says nest_service no longer inherits fallbacks.
- N1 is supported by E1.
- I1 requires abstention because the bundle does not establish a complete
  project-specific patch.

Use E1 for F1 and N1, E2 for F2, and E3 for F3. Include the relevant evidence
even when it limits or contradicts a claim. Use non-empty uncertainty phrases.

EVIDENCE_BUNDLE_JSON:
{bundle}
"""


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _block(text: str, start: str, end: str) -> str:
    left = text.find(start)
    right = text.find(end, left + len(start))
    if left < 0 or right < 0:
        raise ValueError(f"fixture boundary not found: {start!r}")
    return text[left:right].rstrip()


def build_fixture(data: dict[str, Any]) -> tuple[dict[str, Any], str, str]:
    if data.get("session_id") != SESSION_ID:
        raise ValueError("unexpected follow-up session")
    executions = {item.get("tool_call_id"): item for item in data.get("execution", [])}
    expected = {"366090991", "543570296", "584989992"}
    for call_id in expected:
        item = executions.get(call_id)
        if not item or item.get("origin_run_id") != RUN_ID or item.get("status") != "completed":
            raise ValueError(f"follow-up evidence call is not completed in the pinned run: {call_id}")

    server = _tool_content(data, "366090991")
    body = _tool_content(data, "543570296")
    fallback = _tool_content(data, "584989992")
    evidence = [
        {"id": "E1", "tool_call_id": "366090991",
         "source": "https://github.com/tokio-rs/axum/releases/tag/axum-v0.7.0",
         "chunk": "#2", "text": _block(server,
             "https://github.com/tokio-rs/axum/releases/tag/axum-v0.7.0 #2:",
             'Query "hyper Server re-export 0.6"')},
        {"id": "E2", "tool_call_id": "543570296",
         "source": "https://github.com/tokio-rs/axum/blob/main/axum/CHANGELOG.md",
         "chunk": "#32", "text": _block(body,
             "https://github.com/tokio-rs/axum/blob/main/axum/CHANGELOG.md #32:",
             "More: ctx_search")},
        {"id": "E3", "tool_call_id": "584989992",
         "source": "https://github.com/tokio-rs/axum/releases/tag/axum-v0.7.0",
         "chunk": "#2", "text": _block(fallback,
             "https://github.com/tokio-rs/axum/releases/tag/axum-v0.7.0 #2:",
             "- fixed: Don't remove")},
    ]
    final = data["messages"][-1]["content"]
    final_hash = _sha256(final)
    if final_hash != EXPECTED_FINAL_SHA256:
        raise ValueError(f"follow-up final hash mismatch: {final_hash}")
    bundle = {"target": {"run_id": RUN_ID, "session_id": SESSION_ID,
                          "message_index": 19, "sha256": final_hash, "text": final},
              "evidence": evidence}
    bundle_hash = _sha256(json.dumps(bundle, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    if bundle_hash != EXPECTED_BUNDLE_SHA256:
        raise ValueError(f"follow-up bundle hash mismatch: {bundle_hash}")
    return bundle, final_hash, bundle_hash


def _score(review: dict[str, Any]) -> dict[str, Any]:
    expected = {"F1": "contradiction", "F2": "contradiction", "F3": "contradiction"}
    evidence = {"F1": "E1", "F2": "E2", "F3": "E3"}
    findings = {item.get("id"): item for item in review.get("findings", []) if isinstance(item, dict)}
    scores = {item_id: {"classification_correct": findings.get(item_id, {}).get("classification") == value,
                        "evidence_attributed": evidence[item_id] in findings.get(item_id, {}).get("evidence_ids", []),
                        "appropriate_uncertainty": bool(findings.get(item_id, {}).get("uncertainty"))}
              for item_id, value in expected.items()}
    controls = {item.get("id"): item for item in review.get("negative_controls", []) if isinstance(item, dict)}
    insufficiency = review.get("insufficiency_control", {})
    return {"findings": scores,
            "false_positive_controls": [] if controls.get("N1", {}).get("classification") == "supported" else ["N1"],
            "insufficiency_abstained": insufficiency.get("classification") == "insufficient_evidence"
            and insufficiency.get("abstain") is True}


def _validate_review(review: dict[str, Any]) -> str | None:
    """Fail closed on schema drift before any scoring is performed."""
    if set(review) != {"findings", "negative_controls", "insufficiency_control"}:
        return "review root keys do not match the follow-up schema"
    findings = review.get("findings")
    if not isinstance(findings, list) or {item.get("id") for item in findings if isinstance(item, dict)} != {"F1", "F2", "F3"}:
        return "findings must contain exactly F1, F2, and F3"
    if len(findings) != 3 or any(not isinstance(item, dict) for item in findings):
        return "findings contain an invalid item"
    allowed = {"contradiction", "unsupported", "supported", "insufficient_evidence"}
    for item in findings:
        if set(item) != {"id", "detected", "classification", "evidence_ids", "uncertainty", "correction", "abstain"}:
            return f"finding {item.get('id')} keys do not match the schema"
        if item["classification"] not in allowed or not isinstance(item["detected"], bool):
            return f"finding {item.get('id')} has invalid classification or detected type"
        if not isinstance(item["evidence_ids"], list) or not all(isinstance(value, str) for value in item["evidence_ids"]):
            return f"finding {item.get('id')} evidence_ids must be an array of strings"
        if not isinstance(item["uncertainty"], str) or not item["uncertainty"].strip():
            return f"finding {item.get('id')} uncertainty must be non-empty"
        if item["correction"] is not None and not isinstance(item["correction"], str):
            return f"finding {item.get('id')} correction must be a string or null"
        if not isinstance(item["abstain"], bool):
            return f"finding {item.get('id')} abstain must be boolean"
    controls = review.get("negative_controls")
    if not isinstance(controls, list) or len(controls) != 1 or controls[0].get("id") != "N1":
        return "negative_controls must contain exactly N1"
    if set(controls[0]) != {"id", "classification", "evidence_ids"} or controls[0]["classification"] not in allowed:
        return "N1 does not match the schema"
    insufficiency = review.get("insufficiency_control")
    if not isinstance(insufficiency, dict) or set(insufficiency) != {"id", "classification", "abstain", "evidence_ids", "reason"}:
        return "I1 does not match the schema"
    if (insufficiency["id"] != "I1" or insufficiency["classification"] not in allowed
            or insufficiency["classification"] != "insufficient_evidence"
            or insufficiency["abstain"] is not True
            or not isinstance(insufficiency["reason"], str) or not insufficiency["reason"].strip()):
        return "I1 must be insufficient_evidence with abstain=true and a quoted reason"
    return None


def _extract_followup_review(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    try:
        text = payload["choices"][0]["message"]["content"]
        if not isinstance(text, str) or not text.strip():
            return None, "response content is empty"
        if text.lstrip().startswith("```"):
            return None, "Markdown fences are not allowed"
        review = json.loads(text)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        return None, f"response did not contain strict JSON: {exc}"
    if not isinstance(review, dict):
        return None, "review JSON root is not an object"
    error = _validate_review(review)
    return (None, error) if error else (review, None)


def _request_followup(endpoint: str, model: str, bundle: dict[str, Any], seed: int,
                      temperature: float, max_tokens: int, timeout: float) -> tuple[dict[str, Any], float]:
    sdk = _openai_sdk()
    prompt = REVIEW_REQUEST.replace("{bundle}", json.dumps(bundle, ensure_ascii=False, sort_keys=True))
    started = time.monotonic()
    try:
        client = sdk.OpenAI(api_key=os.environ.get("OPENAI_API_KEY") or "not-needed",
                            base_url=normalize_endpoint(endpoint), max_retries=0, timeout=timeout)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": "Return only valid JSON. Use only the supplied evidence."},
                      {"role": "user", "content": prompt}],
            temperature=temperature, max_tokens=max_tokens, seed=seed)
        return _response_payload(response), time.monotonic() - started
    except sdk.APITimeoutError as exc:
        raise ReviewTransportError("api_timeout", "OpenAI SDK request timed out") from exc
    except sdk.APIConnectionError as exc:
        raise ReviewTransportError("api_connection", "OpenAI SDK connection failed") from exc
    except sdk.APIStatusError as exc:
        raise ReviewTransportError("api_status", f"OpenAI SDK API status error ({getattr(exc, 'status_code', None)})") from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--calls", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    data = _load_session(args.session)
    bundle, final_hash, bundle_hash = build_fixture(data)
    result = {"fixture": {"run_id": RUN_ID, "session_id": SESSION_ID,
                           "final_sha256": final_hash, "bundle_sha256": bundle_hash},
              "configuration": {"endpoint": _safe_endpoint(normalize_endpoint(args.endpoint)),
                                 "model": args.model, "calls": args.calls,
                                 "temperature": args.temperature, "max_tokens": args.max_tokens},
              "trials": []}
    for index in range(args.calls):
        trial = {"index": index + 1, "status": "skipped", "review": None, "scoring": None}
        try:
            payload, latency = _request_followup(normalize_endpoint(args.endpoint), args.model, bundle,
                                                 1000 + index, args.temperature, args.max_tokens, args.timeout)
            finish_reason, content, error = _response_metadata(payload)
            trial.update({"finish_reason": finish_reason, "latency_seconds": latency})
            if error or finish_reason == "length" or not isinstance(content, str) or not content.strip():
                trial.update({"status": "incomplete_generation", "error": error or "empty or truncated response"})
            else:
                review, parse_error = _extract_followup_review(payload)
                trial.update({"status": "completed" if review is not None else "response_parse_error",
                              "review": review, "scoring": _score(review) if review is not None else None,
                              "error": parse_error})
                if review is None and isinstance(content, str):
                    trial["raw_content"] = content[:4000]
        except ReviewTransportError as exc:
            trial.update({"status": "transport_error", "error": exc.detail})
        result["trials"].append(trial)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
