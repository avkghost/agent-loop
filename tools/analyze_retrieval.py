#!/usr/bin/env python3
"""Analyze persisted retrieval activity without touching Agent Loop runtime state.

This is intentionally an offline analyzer.  It understands the persisted session
envelope and has a small, context-mode-specific result parser; it is not imported
by the runtime and does not assign quality scores or recommend stopping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


SOURCE_CHUNK_RE = re.compile(r"(https?://[^\s]+?)\s+#(\d+)")
NO_HIT_RE = re.compile(r"no hits?\b", re.IGNORECASE)
ZERO_SOURCE_RE = re.compile(r"across\s+0\s+source(?:s|\(s\))", re.IGNORECASE)
NO_MORE_RE = re.compile(r"no more (?:hits|results)\b", re.IGNORECASE)
EMBEDDED_MARKER_RE = re.compile(r"(?:^|\n)\s*(?:ctx_execute|ctx_search|ctx_batch):", re.IGNORECASE)


class AnalysisError(ValueError):
    """The requested run cannot be analyzed without guessing."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as exc:
        raise AnalysisError(f"session file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise AnalysisError(f"session file is not valid JSON: {exc}") from exc
    except OSError as exc:
        raise AnalysisError(f"could not read session file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise AnalysisError("session root must be a JSON object")
    return data


def _json_arguments(value: Any) -> tuple[dict[str, Any] | None, str | None]:
    if isinstance(value, dict):
        return value, None
    if not isinstance(value, str):
        return None, "arguments are neither an object nor a JSON object string"
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        return None, f"arguments are not valid JSON: {exc.msg}"
    if not isinstance(parsed, dict):
        return None, "arguments JSON is not an object"
    return parsed, None


def _observation_text(value: Any) -> tuple[str, str | None]:
    """Return the context-mode content and an envelope diagnostic.

    MCP results are persisted as serialized envelopes.  Unknown shapes are kept
    as text and classified conservatively rather than discarded.
    """
    if isinstance(value, str):
        raw = value
    else:
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True)
    try:
        envelope = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return raw, "outcome is not a serialized JSON envelope"
    if isinstance(envelope, dict) and isinstance(envelope.get("content"), str):
        return envelope["content"], None
    return raw, "serialized outcome has no string content field"


def _reference_pairs(text: str) -> list[dict[str, Any]]:
    refs = []
    for match in SOURCE_CHUNK_RE.finditer(text):
        source = match.group(1).rstrip(".,;:)]}")
        refs.append({"source": source, "chunk": int(match.group(2))})
    return refs


def _classify(text: str, source: Any) -> dict[str, Any]:
    references = _reference_pairs(text)
    no_hit = bool(NO_HIT_RE.search(text))
    unindexed = bool(source) and bool(ZERO_SOURCE_RE.search(text))
    exhausted = bool(NO_MORE_RE.search(text))
    if references and (no_hit or exhausted):
        # A single MCP observation may contain several query results.  A quoted
        # no-hit line alongside references is not an all-no-hit result.
        primary = "mixed"
    elif unindexed:
        primary = "unindexed_source_label"
    elif exhausted:
        primary = "exhausted_pagination"
    elif no_hit:
        primary = "no_hit"
    elif _reference_pairs(text):
        primary = "results"
    else:
        primary = "unclassified"
    reference_diagnostic = None
    if text.strip() and not references and not (no_hit or exhausted):
        reference_diagnostic = (
            "no recognized source/chunk references; output format may be unsupported"
        )
    return {
        "primary": primary,
        "no_hit": no_hit,
        "unindexed_source_label": unindexed,
        "exhausted_pagination": exhausted,
        "reference_diagnostic": reference_diagnostic,
    }


def _message_cross_check(messages: Any, execution_ids: set[str]) -> dict[str, Any]:
    if not isinstance(messages, list):
        return {"tool_message_ids": [], "unmatched_execution_ids": sorted(execution_ids),
                "unmatched_message_ids": [], "reliable": False,
                "diagnostic": "messages is not a list"}
    message_ids = [m.get("tool_call_id") for m in messages
                   if isinstance(m, dict) and m.get("role") == "tool"
                   and isinstance(m.get("tool_call_id"), str)]
    message_set = set(message_ids)
    return {
        "tool_message_ids": message_ids,
        "unmatched_execution_ids": sorted(execution_ids - message_set),
        "unmatched_message_ids": sorted(message_set - execution_ids),
        "reliable": not (execution_ids - message_set),
    }


def analyze_session(path: str | Path, run_id: str) -> dict[str, Any]:
    """Analyze one run, failing closed where session structure cannot identify it."""
    session_path = Path(path)
    data = _load_json(session_path)
    runs = data.get("runs")
    if not isinstance(runs, list):
        raise AnalysisError("session has no valid runs list; run membership is unavailable")
    matching = [run for run in runs if isinstance(run, dict) and run.get("run_id") == run_id]
    if not matching:
        raise AnalysisError(f"run {run_id} is not present in the session")
    if len(matching) != 1:
        raise AnalysisError(f"run {run_id} appears {len(matching)} times")
    if len(runs) != 1:
        raise AnalysisError(
            "session contains multiple runs, but execution records have no run_id; "
            "tool-call membership cannot be reconstructed safely"
        )

    execution = data.get("execution")
    if not isinstance(execution, list):
        raise AnalysisError("session execution journal is not a list")
    records = []
    for index, record in enumerate(execution):
        if not isinstance(record, dict) or not isinstance(record.get("tool_call_id"), str):
            raise AnalysisError(f"execution record {index} has no usable tool_call_id")
        args, arg_error = _json_arguments(record.get("arguments"))
        output_text, envelope_error = _observation_text(record.get("outcome"))
        persisted_outcome = record.get("outcome", "")
        if not isinstance(persisted_outcome, str):
            persisted_outcome = json.dumps(persisted_outcome, ensure_ascii=False, sort_keys=True)
        tool_name = record.get("tool_name")
        item = {
            "ordinal": index,
            "tool_call_id": record["tool_call_id"],
            "tool_name": tool_name,
            "status": record.get("status"),
            "arguments": args,
            "argument_error": arg_error,
            "content_bytes": len(output_text.encode("utf-8")),
            "persisted_outcome_bytes": len(persisted_outcome.encode("utf-8")),
            "output_sha256": hashlib.sha256(output_text.encode("utf-8")).hexdigest(),
            "envelope_error": envelope_error,
        }
        if tool_name == "mcp_context-mode_ctx_search":
            query_diagnostics = []
            search_args = args if isinstance(args, dict) else {}
            if arg_error:
                query_diagnostics.append(arg_error)
            raw_queries = search_args.get("queries")
            if not isinstance(raw_queries, list):
                query_list = None
                query_diagnostics.append("queries must be a list of strings")
            elif any(not isinstance(query, str) for query in raw_queries):
                query_list = raw_queries
                query_diagnostics.append("queries contains a non-string value")
            else:
                query_list = raw_queries
            source = search_args.get("source")
            if source is not None and not isinstance(source, str):
                query_diagnostics.append("source is not a string")
            references = _reference_pairs(output_text)
            classification = _classify(output_text, source)
            item.update({
                "queries": query_list,
                "query_diagnostics": query_diagnostics,
                "scope": search_args.get("scope"),
                "source": source,
                "offset": search_args.get("offset"),
                "references": references,
                "reference_count": len(references),
                "contains_embedded_prior_output": bool(EMBEDDED_MARKER_RE.search(output_text)),
                "classification": classification,
            })
        records.append(item)

    search = [r for r in records if r["tool_name"] == "mcp_context-mode_ctx_search"]
    message_cross_check = _message_cross_check(
        data.get("messages"), {r["tool_call_id"] for r in records}
    )
    search_ids = {item["tool_call_id"] for item in search}
    missing_search_messages = search_ids & set(message_cross_check["unmatched_execution_ids"])
    if missing_search_messages:
        raise AnalysisError(
            "selected retrieval calls have no matching persisted tool messages; "
            f"boundaries cannot be verified safely: {sorted(missing_search_messages)}"
        )
    pair_counts: Counter[tuple[str, int]] = Counter()
    for item in search:
        pair_counts.update((ref["source"], ref["chunk"]) for ref in item["references"])
    embedded_search = [item for item in search if item["contains_embedded_prior_output"]]
    unflagged_search = [item for item in search if not item["contains_embedded_prior_output"]]
    def pair_counts_for(items: list[dict[str, Any]]) -> Counter[tuple[str, int]]:
        counts: Counter[tuple[str, int]] = Counter()
        for item in items:
            counts.update((ref["source"], ref["chunk"]) for ref in item["references"])
        return counts

    embedded_pair_counts = pair_counts_for(embedded_search)
    unflagged_pair_counts = pair_counts_for(unflagged_search)
    query_counts: Counter[str] = Counter()
    content_counts: Counter[str] = Counter()
    for item in search:
        if not item["query_diagnostics"] and item["queries"] is not None:
            query_counts[json.dumps({"queries": item["queries"], "scope": item["scope"],
                                     "source": item["source"], "offset": item["offset"]},
                                    sort_keys=True, ensure_ascii=False)] += 1
        content_counts[item["output_sha256"]] += 1

    source_sequence = [item["source"] if item["source"] is not None else "<none>" for item in search]
    return {
        "session_path": str(session_path),
        "session_id": data.get("session_id"),
        "run_id": run_id,
        "session_format": data.get("format"),
        "run_status": matching[0].get("status"),
        "run_count": len(runs),
        "tool_call_count": len(records),
        "tool_counts": dict(sorted(Counter(str(r["tool_name"]) for r in records).items())),
        "message_cross_check": message_cross_check,
        "search_count": len(search),
        "search_calls": search,
        "total_persisted_outcome_bytes": sum(item["persisted_outcome_bytes"] for item in search),
        "total_content_bytes": sum(item["content_bytes"] for item in search),
        "reference_count": sum(item["reference_count"] for item in search),
        "unique_references": len(pair_counts),
        "repeated_reference_instances": sum(max(0, count - 1) for count in pair_counts.values()),
        "embedded_reference_count": sum(item["reference_count"] for item in embedded_search),
        "embedded_unique_references": len(embedded_pair_counts),
        "unflagged_reference_count": sum(item["reference_count"] for item in unflagged_search),
        "unflagged_unique_references": len(unflagged_pair_counts),
        "reference_counts": {f"{source} #{chunk}": count for (source, chunk), count in sorted(pair_counts.items())},
        "exact_repeated_query_groups": {query: count for query, count in query_counts.items() if count > 1},
        "exact_repeated_content_groups": {digest: count for digest, count in content_counts.items() if count > 1},
        "classification_counts": dict(sorted(Counter(item["classification"]["primary"] for item in search).items())),
        "no_hit_calls": sum(item["classification"]["no_hit"] for item in search),
        "unindexed_source_calls": sum(item["classification"]["unindexed_source_label"] for item in search),
        "exhausted_pagination_calls": sum(item["classification"]["exhausted_pagination"] for item in search),
        "embedded_output_calls": sum(item["contains_embedded_prior_output"] for item in search),
        "query_diagnostic_count": sum(bool(item["query_diagnostics"]) for item in search),
        "reference_diagnostic_calls": sum(
            bool(item["classification"]["reference_diagnostic"]) for item in search
        ),
        "source_sequence": source_sequence,
        "limitations": [
            "Reference extraction is adapter-specific and recognizes URL-plus-#chunk text only.",
            "Calls whose output contains embedded prior output are flagged; their references are not asserted to be fresh.",
            "The session journal has ordering but no reliable per-call latency or run_id fields.",
            "No quality or efficiency score is derived.",
        ],
    }


def _human(report: dict[str, Any]) -> str:
    lines = [
        f"Run: {report['run_id']}",
        f"Session: {report['session_id']}",
        f"Search calls: {report['search_count']}",
        f"Tool calls: {report['tool_call_count']} ({report['tool_counts']})",
        f"Output: {report['total_persisted_outcome_bytes']} persisted-outcome bytes; "
        f"{report['total_content_bytes']} decoded-content bytes",
        f"Recognized references: {report['reference_count']}; unique pairs: {report['unique_references']}; "
        f"repeated instances: {report['repeated_reference_instances']}",
        f"Embedded-output references: {report['embedded_reference_count']} "
        f"({report['embedded_unique_references']} unique); unflagged references: "
        f"{report['unflagged_reference_count']} ({report['unflagged_unique_references']} unique)",
        f"Classification: {report['classification_counts']}",
        f"Signals: no-hit={report['no_hit_calls']}; unindexed-source={report['unindexed_source_calls']}; "
        f"pagination-exhausted={report['exhausted_pagination_calls']}",
        f"Embedded-output calls flagged: {report['embedded_output_calls']}",
        f"Repeated exact query groups: {len(report['exact_repeated_query_groups'])}",
        f"Repeated exact content groups: {len(report['exact_repeated_content_groups'])}",
        f"Query diagnostics: {report['query_diagnostic_count']}; reference-format diagnostics: "
        f"{report['reference_diagnostic_calls']}",
        f"Message cross-check: {report['message_cross_check']}",
        "",
        "Search calls:",
    ]
    for item in report["search_calls"]:
        classification = item["classification"]["primary"]
        lines.append(
            f"  #{item['ordinal']} {item['tool_call_id']} source={item['source']!r} "
            f"offset={item['offset']!r} refs={item['reference_count']} class={classification}"
            + (" [embedded output flagged]" if item["contains_embedded_prior_output"] else "")
        )
    lines += ["", "Definitions:",
              "  References are URL-plus-#chunk tokens recognized in persisted search output.",
              "  Repeated references count occurrences after the first for each exact (source, chunk) pair.",
              "  Signal counts overlap: one call may be counted in multiple signal categories.",
              "  Unflagged references are not proven fresh primary-source evidence.",
              "  Embedded-output calls are flagged, not silently treated as fresh primary evidence.",
              "  Multi-run sessions without reliable per-call ownership are rejected."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze retrieval calls in a persisted Agent Loop session")
    parser.add_argument("session_json", type=Path, help="path to a persisted session JSON file")
    parser.add_argument("--run-id", required=True, help="exact run identifier to analyze")
    parser.add_argument("--json", action="store_true", dest="as_json", help="emit machine-readable JSON")
    args = parser.parse_args(argv)
    try:
        report = analyze_session(args.session_json, args.run_id)
    except AnalysisError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.as_json:
        print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        print(_human(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
