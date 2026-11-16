import json
from pathlib import Path

import pytest

from tools.analyze_retrieval import AnalysisError, analyze_session


def _session(tmp_path: Path, *, runs, execution, messages=None):
    path = tmp_path / "session.json"
    path.write_text(json.dumps({
        "format": 5,
        "session_id": "session-test",
        "runs": runs,
        "execution": execution,
        "messages": messages if messages is not None else [
            {"role": "tool", "tool_call_id": record["tool_call_id"], "content": record["outcome"]}
            for record in execution
        ],
    }), encoding="utf-8")
    return path


def _record(call_id, args, content, name="mcp_context-mode_ctx_search"):
    return {"tool_call_id": call_id, "tool_name": name, "arguments": args,
            "status": "completed", "outcome": json.dumps({"content": content, "isError": False})}


def _run(run_id="run_" + "a" * 32):
    return [{"run_id": run_id, "status": "completed_unverified"}]


def test_run_identification_and_metrics(tmp_path):
    run_id = "run_" + "a" * 32
    content = "https://docs.example/axum #1: result\nhttps://docs.example/axum #1: repeated"
    path = _session(tmp_path, runs=_run(run_id), execution=[
        _record("a", {"queries": ["q"], "source": "docs"}, content),
        _record("b", {"queries": ["q2"], "source": "docs", "offset": 5}, "No hits in scope \"worktree\" across 0 source(s)."),
    ])
    report = analyze_session(path, run_id)
    assert report["search_count"] == 2
    assert report["reference_count"] == 2
    assert report["unique_references"] == 1
    assert report["repeated_reference_instances"] == 1
    assert report["classification_counts"] == {"results": 1, "unindexed_source_label": 1}
    assert report["no_hit_calls"] == 1
    assert report["unindexed_source_calls"] == 1
    assert report["search_calls"][1]["offset"] == 5


def test_embedded_and_unflagged_reference_breakdowns_are_separate(tmp_path):
    run_id = "run_" + "a" * 32
    path = _session(tmp_path, runs=_run(run_id), execution=[
        _record("a", {"queries": ["q"]},
                "ctx_execute: prior\nhttps://docs.example/a #1: old"),
        _record("b", {"queries": ["q2"]}, "https://docs.example/a #1: result\nhttps://docs.example/a #2: result"),
    ])
    report = analyze_session(path, run_id)
    assert report["reference_count"] == 3
    assert report["unique_references"] == 2
    assert report["embedded_reference_count"] == 1
    assert report["embedded_unique_references"] == 1
    assert report["unflagged_reference_count"] == 2
    assert report["unflagged_unique_references"] == 2


def test_multiple_runs_fail_closed(tmp_path):
    run_id = "run_" + "a" * 32
    path = _session(tmp_path, runs=_run(run_id) + [{"run_id": "run_" + "b" * 32}], execution=[])
    with pytest.raises(AnalysisError, match="multiple runs"):
        analyze_session(path, run_id)


def test_multiple_runs_select_only_records_owned_by_requested_run(tmp_path):
    run_id = "run_" + "a" * 32
    other_run_id = "run_" + "b" * 32
    path = _session(tmp_path, runs=_run(run_id) + [{"run_id": other_run_id}], execution=[
        {**_record("owned", {"queries": ["q"]}, "https://docs.example/a #1: owned"),
         "origin_run_id": run_id},
        {**_record("foreign", {"queries": ["q"]}, "https://docs.example/b #2: foreign"),
         "origin_run_id": other_run_id},
    ])
    report = analyze_session(path, run_id)
    assert report["tool_call_count"] == 1
    assert report["search_count"] == 1
    assert report["selected_execution_count"] == 1
    assert report["owned_execution_count"] == 1
    assert report["unowned_execution_count"] == 0
    assert report["foreign_execution_count"] == 1
    assert report["search_calls"][0]["tool_call_id"] == "owned"


def test_multiple_runs_with_legacy_execution_record_fail_closed(tmp_path):
    run_id = "run_" + "a" * 32
    path = _session(tmp_path, runs=_run(run_id) + [{"run_id": "run_" + "b" * 32}], execution=[
        _record("legacy", {"queries": ["q"]}, "https://docs.example/a #1: ambiguous"),
    ])
    with pytest.raises(AnalysisError, match="without origin_run_id"):
        analyze_session(path, run_id)


def test_missing_run_and_malformed_data_fail(tmp_path):
    path = _session(tmp_path, runs=_run(), execution=[])
    with pytest.raises(AnalysisError, match="not present"):
        analyze_session(path, "run_" + "b" * 32)
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    with pytest.raises(AnalysisError, match="not valid JSON"):
        analyze_session(bad, "run_" + "a" * 32)


def test_unindexed_and_exhausted_outputs_are_classified(tmp_path):
    run_id = "run_" + "a" * 32
    path = _session(tmp_path, runs=_run(run_id), execution=[
        _record("a", {"queries": ["q"], "source": "https://docs.invalid/subpath"},
                'No hits in scope "worktree" across 0 source(s).'),
        _record("b", {"queries": ["q"], "source": "docs", "offset": 10},
                'No more hits past offset 10 in scope "worktree" across 1 source(s).'),
    ])
    report = analyze_session(path, run_id)
    assert report["classification_counts"] == {"exhausted_pagination": 1, "unindexed_source_label": 1}
    assert report["search_calls"][0]["classification"]["unindexed_source_label"]


def test_references_plus_quoted_no_hit_is_mixed_not_no_hit(tmp_path):
    run_id = "run_" + "a" * 32
    path = _session(tmp_path, runs=_run(run_id), execution=[
        _record("a", {"queries": ["one", "two"]},
                'Query "one": No hits in scope.\nQuery "two": https://docs.example/a #4: hit'),
    ])
    report = analyze_session(path, run_id)
    assert report["classification_counts"] == {"mixed": 1}
    assert report["no_hit_calls"] == 1


def test_malformed_queries_and_reference_format_are_diagnostic(tmp_path):
    run_id = "run_" + "a" * 32
    path = _session(tmp_path, runs=_run(run_id), execution=[
        _record("a", {"queries": ["ok", 3], "source": 42}, "result without a recognized source format"),
    ])
    report = analyze_session(path, run_id)
    call = report["search_calls"][0]
    assert call["query_diagnostics"]
    assert "non-string" in call["query_diagnostics"][0]
    assert call["classification"]["primary"] == "unclassified"
    assert call["classification"]["reference_diagnostic"]
    assert report["query_diagnostic_count"] == 1
    assert report["reference_diagnostic_calls"] == 1


def test_repeated_text_and_embedded_prior_output_are_reported(tmp_path):
    run_id = "run_" + "a" * 32
    content = "ctx_execute: #0: prior output\nhttps://docs.example/axum #3: old"
    path = _session(tmp_path, runs=_run(run_id), execution=[
        _record("a", {"queries": ["q"]}, content),
        _record("b", {"queries": ["q"]}, content),
    ])
    report = analyze_session(path, run_id)
    assert report["embedded_output_calls"] == 2
    assert len(report["exact_repeated_content_groups"]) == 1
    assert report["reference_count"] == 2
    assert report["unique_references"] == 1


def test_message_cross_check_reports_unmatched_ids(tmp_path):
    run_id = "run_" + "a" * 32
    execution = [_record("a", {"queries": ["q"]}, "plain")]
    path = _session(tmp_path, runs=_run(run_id), execution=execution, messages=[])
    with pytest.raises(AnalysisError, match="boundaries cannot be verified"):
        analyze_session(path, run_id)
