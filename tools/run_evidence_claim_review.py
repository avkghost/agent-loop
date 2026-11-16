#!/usr/bin/env python3
"""Standalone, opt-in runner for the Evidence-to-Claim Review Pilot.

This module is deliberately outside Agent Loop.  It loads an existing session,
constructs a verified immutable review fixture, and optionally sends five
stateless JSON-only requests to an explicitly configured OpenAI-compatible chat
endpoint.  It never invokes Agent Loop or retrieval tools.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from urllib.parse import urlsplit, urlunsplit
from pathlib import Path
from typing import Any


EXPECTED_FINAL_SHA256 = "941e0664e1b26cb13cb06115c0a5fed8d8bc9e8f334ef30141c6250de9208fd9"
EXPECTED_BUNDLE_SHA256 = "09f324307c26d108c77f24409131ac864c324dd14be25d710a4ee8c419d265d1"
DEFAULT_RUN_ID = "run_d2ffcd5e2ee14c53915925de6425369c"
DEFAULT_SESSION = Path.home() / ".agent_loop" / "sessions" / "498653a0-8a5e-4f47-8bf1-15a0c47622a5.json"

SYSTEM_INSTRUCTION = """You are an independent evidence-to-claim reviewer.
Use only the supplied evidence bundle and target answer. Do not use prior
knowledge, retrieval, tools, or external sources. A source may support only
what its exact text establishes. Classify each claim independently. When the
bundle does not establish an exact API detail, abstain; never invent a
replacement path. Return only valid JSON matching the requested schema.
Detection and correction are separate: a correction may be an explicit
abstention."""

REVIEW_REQUEST = """Review the target answer against the evidence bundle.

Return exactly this JSON shape:
{
  "findings": [
    {"id":"F1|F2|F3", "detected":true,
     "classification":"contradiction|unsupported|supported|insufficient_evidence",
     "evidence_ids":["E1"], "uncertainty":"...",
     "correction":"...", "abstain":false}
  ],
  "negative_controls": [
    {"id":"N1|N2|N3", "classification":"contradiction|unsupported|supported|insufficient_evidence",
     "evidence_ids":["E4"]}
  ],
  "insufficiency_control":
    {"id":"I1", "classification":"insufficient_evidence", "abstain":true,
     "evidence_ids":[], "reason":"..."}
}

Evaluate these exact target claims; do not assume any classification:
F1: "Removed types/traits: FromRequestParts, FromRequest, HandlerService,
HandlerWithoutStateExt, Handler, LayeredFuture, Layered, MethodRouter, Next,
RequestExt, RouteFuture, Route, Router"
F2: "No code changes required; this is an internal implementation detail."
F3: "use axum_extra::extract::TypedHeader;"
N1: "axum::serve(listener, app).await.unwrap();"
N2: "TypedHeader moved to axum-extra crate."
N3: "axum::body::Body replaces the removed hyper::Body re-export."
I1: "The evidence proves the exact TypedHeader import path."

Use the supplied evidence IDs to attribute support or limitation. N1-N3 are
controls, not pre-labeled answers. I1 has no evidence item establishing an
exact import path, so it may require abstention.

Do not add fields outside the schema and do not provide an invented import path.

EVIDENCE_BUNDLE_JSON:
{bundle}
"""


class RunnerError(ValueError):
    """Fixture or configuration error; no model request was made."""


class ReviewTransportError(RuntimeError):
    """A typed transport/API failure that must not be scored as a review."""

    def __init__(self, category: str, detail: str):
        super().__init__(detail)
        self.category = category
        self.detail = detail


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_endpoint(endpoint: str) -> str:
    """Display an endpoint without query credentials or userinfo."""
    parts = urlsplit(endpoint)
    host = parts.hostname or ""
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def normalize_endpoint(endpoint: str) -> str:
    """Return an OpenAI SDK base_url from /v1 or /v1/chat/completions."""
    if not isinstance(endpoint, str) or not endpoint or any(char.isspace() for char in endpoint):
        raise RunnerError("endpoint must be a URL without whitespace or Markdown formatting")
    if any(marker in endpoint for marker in ("```", "[", "](", "<", ">")):
        raise RunnerError("endpoint must be a plain URL, not Markdown-formatted text")
    parts = urlsplit(endpoint)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise RunnerError("endpoint must be an absolute http(s) URL")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise RunnerError("endpoint must not contain credentials, query parameters, or fragments")
    path = parts.path.rstrip("/")
    suffix = "/chat/completions"
    if path.endswith(suffix):
        path = path[:-len(suffix)].rstrip("/")
    if not path.endswith("/v1"):
        raise RunnerError("endpoint must end with /v1 or /v1/chat/completions")
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def _openai_sdk():
    try:
        import openai
    except ImportError as exc:
        raise RunnerError("the openai package is required for the pilot runner") from exc
    return openai


def _load_session(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RunnerError(f"session file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RunnerError(f"session file is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise RunnerError("session root must be an object")
    return data


def _tool_content(data: dict[str, Any], call_id: str) -> str:
    record = next((item for item in data.get("execution", [])
                   if isinstance(item, dict) and item.get("tool_call_id") == call_id), None)
    if record is None:
        raise RunnerError(f"required evidence tool call is missing: {call_id}")
    try:
        outcome = json.loads(record["outcome"])
        content = outcome["content"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RunnerError(f"tool call {call_id} has no usable persisted content") from exc
    if not isinstance(content, str):
        raise RunnerError(f"tool call {call_id} content is not text")
    return content


def build_fixture(data: dict[str, Any]) -> tuple[dict[str, Any], str, str]:
    """Build the canonical fixture from persisted observations without writing it."""
    messages = data.get("messages")
    if not isinstance(messages, list) or len(messages) <= 72:
        raise RunnerError("session does not contain the canonical final message at index 72")
    final = messages[72].get("content") if isinstance(messages[72], dict) else None
    if not isinstance(final, str):
        raise RunnerError("canonical final answer is not text")
    final_hash = _sha256_text(final)
    if final_hash != EXPECTED_FINAL_SHA256:
        raise RunnerError(f"final answer hash mismatch: {final_hash}")

    generic = _tool_content(data, "394757186")
    s1 = generic.find("- breaking: The following types/traits")
    s2 = generic.find("- breaking: axum no longer re-exports", s1)
    s3 = generic.find('Query "B type param":', s2)
    typed = _tool_content(data, "951879790")
    s4 = typed.find("- breaking: `TypedHeader`")
    s5 = typed.find('Query "typed header":', s4)
    serve = _tool_content(data, "162183842")
    s6 = serve.find("The “Hello, World!” of axum is:")
    s7 = serve.find("Note using", s6)
    if min(s1, s2, s3, s4, s5, s6, s7) < 0:
        raise RunnerError("canonical evidence boundaries were not found")

    evidence = [
        {"id": "E1", "tool_call_id": "394757186", "source": "https://github.com/tokio-rs/axum/releases/tag/axum-v0.7.0", "chunk": "#2", "text": generic[s1:s2].rstrip()},
        {"id": "E2", "tool_call_id": "394757186", "source": "https://github.com/tokio-rs/axum/releases/tag/axum-v0.7.0", "chunk": "#2", "text": generic[s2:s3].rstrip()},
        {"id": "E3", "tool_call_id": "951879790", "source": "https://github.com/tokio-rs/axum/releases/tag/axum-v0.7.0", "chunk": "#2", "text": typed[s4:s5].rstrip()},
        {"id": "E4", "tool_call_id": "162183842", "source": "https://docs.rs/axum/0.7.0/axum/", "chunk": "#4", "text": serve[s6:s7].rstrip()},
    ]
    bundle = {
        "target": {"run_id": DEFAULT_RUN_ID, "session_id": data.get("session_id"),
                    "message_index": 72, "sha256": final_hash, "text": final},
        "evidence": evidence,
    }
    canonical = json.dumps(bundle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    bundle_hash = _sha256_text(canonical)
    if bundle_hash != EXPECTED_BUNDLE_SHA256:
        raise RunnerError(f"evidence bundle hash mismatch: {bundle_hash}")
    return bundle, final_hash, bundle_hash


def _classification_score(review: dict[str, Any]) -> dict[str, Any]:
    expected = {"F1": "contradiction", "F2": "unsupported", "F3": "insufficient_evidence"}
    findings = {item.get("id"): item for item in review.get("findings", [])
                if isinstance(item, dict)}
    scores = {}
    for item_id, category in expected.items():
        item = findings.get(item_id, {})
        scores[item_id] = {
            "detected": bool(item.get("detected")),
            "classification_correct": item.get("classification") == category,
            "evidence_attributed": item_id.replace("F", "E") in item.get("evidence_ids", []),
            "appropriate_uncertainty": (bool(item.get("abstain")) if item_id == "F3"
                                         else isinstance(item.get("uncertainty"), str)
                                         and bool(item.get("uncertainty"))),
            "correction_accuracy": _correction_is_bounded(item_id, item),
        }
    controls = {item.get("id"): item for item in review.get("negative_controls", [])
                if isinstance(item, dict)}
    false_positives = [item_id for item_id in ("N1", "N2", "N3")
                       if controls.get(item_id, {}).get("classification") != "supported"]
    insufficiency = review.get("insufficiency_control", {})
    return {"findings": scores, "false_positive_controls": false_positives,
            "insufficiency_abstained": insufficiency.get("classification") == "insufficient_evidence"
            and insufficiency.get("abstain") is True}


def _correction_is_bounded(item_id: str, item: dict[str, Any]) -> bool:
    correction = item.get("correction")
    if not isinstance(correction, str) or not correction.strip():
        return False
    lowered = correction.lower()
    if item_id == "F1":
        return "no longer generic" in lowered or "b type" in lowered
    if item_id == "F2":
        return "not established" in lowered or "depends" in lowered or "unsupported" in lowered
    if item_id == "F3":
        return bool(item.get("abstain")) and "axum_extra::extract::typedheader" not in lowered
    return False


def _response_payload(response: Any) -> dict[str, Any]:
    """Convert an SDK response to JSON-safe data while retaining raw fields."""
    if isinstance(response, dict):
        return response
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "to_dict"):
        return response.to_dict()
    raise ReviewTransportError("response_parsing", "SDK response is not JSON-serializable")


def _response_metadata(payload: dict[str, Any]) -> tuple[str | None, Any, str | None]:
    try:
        choice = payload["choices"][0]
        message = choice["message"]
        content = message.get("content") if isinstance(message, dict) else None
        return choice.get("finish_reason"), content, None
    except (KeyError, IndexError, TypeError) as exc:
        return None, None, f"response shape is incomplete: {type(exc).__name__}"


def _request(endpoint: str, api_key: str | None, model: str, bundle: dict[str, Any],
             seed: int | None, temperature: float, max_tokens: int, timeout: float) -> tuple[dict[str, Any], float]:
    """Make exactly one SDK request with no response_format or conversation state."""
    sdk = _openai_sdk()
    prompt = REVIEW_REQUEST.replace("{bundle}", json.dumps(bundle, ensure_ascii=False, sort_keys=True))
    started = time.monotonic()
    try:
        client = sdk.OpenAI(
            api_key=api_key or "not-needed",
            base_url=normalize_endpoint(endpoint),
            max_retries=0,
            timeout=timeout,
        )
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": SYSTEM_INSTRUCTION},
                         {"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            kwargs["seed"] = seed
        response = client.chat.completions.create(**kwargs)
        payload = _response_payload(response)
    except sdk.APITimeoutError as exc:
        raise ReviewTransportError("api_timeout", "OpenAI SDK request timed out") from exc
    except sdk.APIConnectionError as exc:
        raise ReviewTransportError("api_connection", "OpenAI SDK connection failed") from exc
    except sdk.APIStatusError as exc:
        status = getattr(exc, "status_code", None)
        detail = f"OpenAI SDK API status error{f' ({status})' if status else ''}"
        raise ReviewTransportError("api_status", detail) from exc
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ReviewTransportError("response_parsing", f"SDK response parsing failed: {type(exc).__name__}") from exc
    return payload, time.monotonic() - started


def _extract_review(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    try:
        text = payload["choices"][0]["message"]["content"]
        review = json.loads(text)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        return None, f"response did not contain structured review JSON: {exc}"
    if not isinstance(review, dict):
        return None, "review JSON root is not an object"
    return review, None


def run_pilot(args: argparse.Namespace) -> dict[str, Any]:
    data = _load_session(args.session)
    bundle, final_hash, bundle_hash = build_fixture(data)
    base_url = normalize_endpoint(args.endpoint)
    if not 1 <= args.calls <= 5:
        raise RunnerError("--calls must be between 1 and 5")
    result: dict[str, Any] = {
        "fixture": {"final_sha256": final_hash, "bundle_sha256": bundle_hash,
                    "run_id": DEFAULT_RUN_ID, "session_id": data.get("session_id")},
        "configuration": {"endpoint": _safe_endpoint(base_url), "model": args.model,
                          "temperature": args.temperature, "max_tokens": args.max_tokens,
                          "timeout_seconds": args.timeout, "seed_start": args.seed_start,
                          "calls": args.calls, "system_instruction_sha256": _sha256_text(SYSTEM_INSTRUCTION),
                          "api_key_env": args.api_key_env},
        "trials": [],
    }
    if args.dry_run:
        result["dry_run"] = True
        result["note"] = "No model request was made."
        return result
    api_key = os.environ.get(args.api_key_env)
    for index in range(args.calls):
        seed = args.seed_start + index if args.seed_start is not None else None
        trial: dict[str, Any] = {"index": index + 1, "seed": seed, "status": "skipped",
                                 "review": None, "scoring": None, "raw_response": None,
                                 "usage": None, "error": None, "latency_seconds": None}
        try:
            payload, latency = _request(base_url, api_key, args.model, bundle, seed,
                                        args.temperature, args.max_tokens, args.timeout)
            finish_reason, content, metadata_error = _response_metadata(payload)
            trial.update({"raw_response": payload, "finish_reason": finish_reason,
                          "usage": payload.get("usage") if isinstance(payload, dict) else None,
                          "latency_seconds": latency})
            if metadata_error:
                trial.update({"status": "incomplete_response", "error": metadata_error})
            elif finish_reason == "length" or not isinstance(content, str) or not content.strip():
                trial.update({"status": "incomplete_generation",
                              "error": "response content was empty or truncated; review was not scored"})
            else:
                review, parse_error = _extract_review(payload)
                trial.update({"status": "completed" if review is not None else "response_parse_error",
                              "review": review,
                              "scoring": _classification_score(review) if review is not None else None,
                              "error": parse_error})
        except ReviewTransportError as exc:
            trial.update({"status": "transport_error", "error_category": exc.category,
                          "error": exc.detail})
        result["trials"].append(trial)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the offline Evidence-to-Claim Review Pilot")
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--endpoint", required=True,
                        help="OpenAI-compatible base URL (/v1) or Chat Completions URL (/v1/chat/completions)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="environment variable name; value is never printed")
    parser.add_argument("--output", type=Path, help="optional JSON result path")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--calls", type=int, default=5, help="number of independent calls (1-5; default: 5)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run_pilot(args)
    except RunnerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    serialized = json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True)
    if args.output:
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
