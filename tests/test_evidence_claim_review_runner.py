import json
from argparse import Namespace
from pathlib import Path

import pytest

import tools.run_evidence_claim_review as runner
from tools.run_evidence_claim_review import (
    EXPECTED_BUNDLE_SHA256,
    EXPECTED_FINAL_SHA256,
    _classification_score,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def model_dump(self):
        return self.payload


class FakeSDK:
    class APIStatusError(Exception):
        status_code = 400

    class APITimeoutError(Exception):
        pass

    class APIConnectionError(Exception):
        pass

    responses = []
    clients = []
    requests = []
    failure = None

    class _Completions:
        def __init__(self, client):
            self.client = client

        def create(self, **kwargs):
            FakeSDK.requests.append(kwargs)
            if FakeSDK.failure:
                raise FakeSDK.failure
            return FakeSDK.responses.pop(0)

    class _Chat:
        def __init__(self, client):
            self.completions = FakeSDK._Completions(client)

    class _Client:
        def __init__(self, **kwargs):
            FakeSDK.clients.append(kwargs)
            self.chat = FakeSDK._Chat(self)

    OpenAI = _Client


def _args(**overrides):
    values = dict(session=Path("unused"), endpoint="http://localhost:1234/v1/chat/completions",
                  model="fake-model", api_key_env="PILOT_KEY", output=None,
                  temperature=0.0, max_tokens=1000, timeout=3.0, seed_start=10,
                  calls=5, dry_run=False)
    values.update(overrides)
    return Namespace(**values)


def _valid_payload(finish_reason="stop", content=None):
    if content is None:
        content = json.dumps({"findings": [], "negative_controls": [], "insufficiency_control": {}})
    return {"id": "r", "choices": [{"finish_reason": finish_reason,
            "message": {"role": "assistant", "content": content}}],
            "usage": {"completion_tokens": 3}}


@pytest.fixture(autouse=True)
def reset_fake_sdk(monkeypatch):
    FakeSDK.responses = []
    FakeSDK.clients = []
    FakeSDK.requests = []
    FakeSDK.failure = None
    monkeypatch.setattr(runner, "_openai_sdk", lambda: FakeSDK)


def test_scoring_keeps_detection_and_correction_separate():
    review = {
        "findings": [
            {"id": "F1", "detected": True, "classification": "contradiction",
             "evidence_ids": ["E1"], "uncertainty": "direct", "correction": "The evidence does not establish that the types were removed.", "abstain": False},
            {"id": "F2", "detected": True, "classification": "unsupported",
             "evidence_ids": ["E1"], "uncertainty": "impact is not established", "correction": "The project-specific impact is not established.", "abstain": False},
            {"id": "F3", "detected": True, "classification": "insufficient_evidence",
             "evidence_ids": ["E3"], "uncertainty": "path absent", "correction": "The exact import path is not established.", "abstain": True},
            {"id": "F4", "detected": True, "classification": "supported",
             "evidence_ids": ["E1"], "uncertainty": "direct", "correction": "The B type parameter is no longer generic.", "abstain": False},
        ],
        "negative_controls": [{"id": "N1", "classification": "supported"},
                              {"id": "N2", "classification": "supported"},
                              {"id": "N3", "classification": "supported"}],
        "insufficiency_control": {"id": "I1", "classification": "insufficient_evidence", "abstain": True},
    }
    score = _classification_score(review)
    assert all(item["detected"] for item in score["findings"].values())
    assert all(item["classification_correct"] for item in score["findings"].values())
    assert all(item["correction_accuracy"] for item in score["findings"].values())
    assert score["false_positive_controls"] == []
    assert score["insufficiency_abstained"]


def test_invented_typed_header_path_is_not_a_correction():
    score = _classification_score({
        "findings": [{"id": "F3", "detected": True, "classification": "insufficient_evidence",
                      "evidence_ids": ["E3"], "uncertainty": "unknown",
                      "correction": "use axum_extra::extract::TypedHeader", "abstain": False}],
        "negative_controls": [], "insufficiency_control": {},
    })
    assert not score["findings"]["F3"]["correction_accuracy"]


def test_abstention_may_quote_typed_header_path_when_denied():
    score = _classification_score({
        "findings": [{"id": "F3", "detected": True, "classification": "insufficient_evidence",
                      "evidence_ids": ["E3"], "uncertainty": "path absent",
                      "correction": "The evidence does not establish the exact path axum_extra::extract::TypedHeader.",
                      "abstain": True}],
        "negative_controls": [], "insufficiency_control": {},
    })
    assert score["findings"]["F3"]["correction_accuracy"]


def test_supported_b_generic_claim_does_not_require_correction():
    score = _classification_score({
        "findings": [{"id": "F4", "detected": True, "classification": "supported",
                      "evidence_ids": ["E1"], "uncertainty": "direct",
                      "correction": "", "abstain": False}],
        "negative_controls": [], "insufficiency_control": {},
    })
    assert score["findings"]["F4"]["correction_accuracy"]


def test_supported_b_generic_claim_may_use_null_correction():
    score = _classification_score({
        "findings": [{"id": "F4", "detected": True, "classification": "supported",
                      "evidence_ids": ["E1"], "uncertainty": "direct evidence",
                      "correction": None, "abstain": False}],
        "negative_controls": [], "insufficiency_control": {},
    })
    assert score["findings"]["F4"]["correction_accuracy"]


def test_unsupported_correction_may_state_that_impact_is_not_established():
    score = _classification_score({
        "findings": [{"id": "F2", "detected": True, "classification": "unsupported",
                      "evidence_ids": ["E1"], "uncertainty": "impact is not established",
                      "correction": "The evidence does not establish that no code changes are required.",
                      "abstain": False}],
        "negative_controls": [], "insufficiency_control": {},
    })
    assert score["findings"]["F2"]["correction_accuracy"]


def test_dry_run_configuration_is_explicit():
    assert EXPECTED_FINAL_SHA256 == "941e0664e1b26cb13cb06115c0a5fed8d8bc9e8f334ef30141c6250de9208fd9"
    assert EXPECTED_BUNDLE_SHA256 == "09f324307c26d108c77f24409131ac864c324dd14be25d710a4ee8c419d265d1"


def test_target_provenance_requires_the_pinned_run_and_rejects_explicit_mismatch():
    base = {"runs": [{"run_id": runner.DEFAULT_RUN_ID}], "execution": [
        {"tool_call_id": "394757186", "origin_run_id": None, "status": "completed",
         "tool_name": "mcp_context-mode_ctx_search"},
    ]}
    runner._validate_target_provenance(base)

    with pytest.raises(runner.RunnerError, match="pinned target run"):
        runner._validate_target_provenance({"runs": [], "execution": []})

    mismatch = {"runs": [{"run_id": runner.DEFAULT_RUN_ID}], "execution": [
        {"tool_call_id": "394757186", "origin_run_id": "run_" + "a" * 32, "status": "completed",
         "tool_name": "mcp_context-mode_ctx_search"},
    ]}
    with pytest.raises(runner.RunnerError, match="different run"):
        runner._validate_target_provenance(mismatch)

    incomplete = {"runs": [{"run_id": runner.DEFAULT_RUN_ID}], "execution": [
        {"tool_call_id": "394757186", "origin_run_id": None, "status": "unknown",
         "tool_name": "mcp_context-mode_ctx_search"},
    ]}
    with pytest.raises(runner.RunnerError, match="not completed"):
        runner._validate_target_provenance(incomplete)

    wrong_tool = {"runs": [{"run_id": runner.DEFAULT_RUN_ID}], "execution": [
        {"tool_call_id": "394757186", "origin_run_id": None, "status": "completed",
         "tool_name": "run_command"},
    ]}
    with pytest.raises(runner.RunnerError, match="unexpected tool name"):
        runner._validate_target_provenance(wrong_tool)

    duplicate = {"runs": [{"run_id": runner.DEFAULT_RUN_ID}], "execution": [
        {"tool_call_id": "394757186", "origin_run_id": None, "status": "completed",
         "tool_name": "mcp_context-mode_ctx_search"},
        {"tool_call_id": "394757186", "origin_run_id": None, "status": "completed",
         "tool_name": "mcp_context-mode_ctx_search"},
    ]}
    with pytest.raises(runner.RunnerError, match="duplicate evidence tool call ID"):
        runner._validate_target_provenance(duplicate)


def test_tool_content_rejects_error_or_malformed_evidence_envelopes():
    with pytest.raises(runner.RunnerError, match="not a successful evidence envelope"):
        runner._tool_content({"execution": [{"tool_call_id": "c", "outcome":
                                               json.dumps({"content": "error", "isError": True})}]}, "c")
    with pytest.raises(runner.RunnerError, match="no usable persisted content"):
        runner._tool_content({"execution": [{"tool_call_id": "c", "outcome": "not-json"}]}, "c")


def test_source_chunk_marker_must_match_declared_evidence():
    runner._require_source_chunk(
        "https://docs.rs/axum/0.7.0/axum/ #4: evidence",
        "https://docs.rs/axum/0.7.0/axum/", 4)
    with pytest.raises(runner.RunnerError, match="source/chunk marker"):
        runner._require_source_chunk(
            "https://docs.rs/axum/0.7.0/axum/ #3: evidence",
            "https://docs.rs/axum/0.7.0/axum/", 4)


def test_endpoint_normalization_and_markdown_rejection():
    assert runner.normalize_endpoint("http://127.0.0.1:1234/v1") == "http://127.0.0.1:1234/v1"
    assert runner.normalize_endpoint("http://127.0.0.1:1234/v1/chat/completions") == "http://127.0.0.1:1234/v1"
    with pytest.raises(runner.RunnerError):
        runner.normalize_endpoint("[http://127.0.0.1:1234/v1](url)")


def test_five_requests_are_independent_and_sdk_parameters_are_controlled(monkeypatch):
    FakeSDK.responses = [FakeResponse(_valid_payload()) for _ in range(5)]
    monkeypatch.setattr(runner, "_load_session", lambda path: {"session_id": "s"})
    monkeypatch.setattr(runner, "build_fixture", lambda data: ({"evidence": []}, "f", "b"))
    result = runner.run_pilot(_args())
    assert [trial["status"] for trial in result["trials"]] == ["completed"] * 5
    assert len(FakeSDK.clients) == 5
    assert all(client["max_retries"] == 0 for client in FakeSDK.clients)
    assert all(client["base_url"] == "http://localhost:1234/v1" for client in FakeSDK.clients)
    assert [request["seed"] for request in FakeSDK.requests] == [10, 11, 12, 13, 14]
    assert all("response_format" not in request for request in FakeSDK.requests)
    assert len({id(request["messages"]) for request in FakeSDK.requests}) == 5
    assert len({request["messages"] [0]["content"] for request in FakeSDK.requests}) == 1


def test_call_count_is_limited_to_one_through_five(monkeypatch):
    monkeypatch.setattr(runner, "_load_session", lambda path: {"session_id": "s"})
    monkeypatch.setattr(runner, "build_fixture", lambda data: ({"evidence": []}, "f", "b"))
    with pytest.raises(runner.RunnerError, match="between 1 and 5"):
        runner.run_pilot(_args(calls=0))
    with pytest.raises(runner.RunnerError, match="between 1 and 5"):
        runner.run_pilot(_args(calls=6))


def test_length_empty_reasoning_response_is_incomplete_and_unscored(monkeypatch):
    payload = _valid_payload(finish_reason="length", content=None)
    payload["choices"][0]["message"]["content"] = ""
    payload["choices"][0]["message"]["reasoning_content"] = "long internal reasoning"
    payload["usage"] = {"completion_tokens": 4095, "completion_tokens_details": {"reasoning_tokens": 4095}}
    FakeSDK.responses = [FakeResponse(payload)]
    monkeypatch.setattr(runner, "_load_session", lambda path: {"session_id": "s"})
    monkeypatch.setattr(runner, "build_fixture", lambda data: ({"evidence": []}, "f", "b"))
    trial = runner.run_pilot(_args(calls=1))["trials"][0]
    assert trial["status"] == "incomplete_generation"
    assert trial["finish_reason"] == "length"
    assert trial["usage"]["completion_tokens_details"]["reasoning_tokens"] == 4095
    assert trial["scoring"] is None


def test_malformed_json_is_parse_error_not_incorrect_review(monkeypatch):
    FakeSDK.responses = [FakeResponse(_valid_payload(content="not json"))]
    monkeypatch.setattr(runner, "_load_session", lambda path: {"session_id": "s"})
    monkeypatch.setattr(runner, "build_fixture", lambda data: ({"evidence": []}, "f", "b"))
    trial = runner.run_pilot(_args(calls=1))["trials"][0]
    assert trial["status"] == "response_parse_error"
    assert trial["scoring"] is None


def test_a_single_json_code_fence_is_accepted(monkeypatch):
    payload = _valid_payload(content="```json\n{}\n```")
    FakeSDK.responses = [FakeResponse(payload)]
    monkeypatch.setattr(runner, "_load_session", lambda path: {"session_id": "s"})
    monkeypatch.setattr(runner, "build_fixture", lambda data: ({"evidence": []}, "f", "b"))
    trial = runner.run_pilot(_args(calls=1))["trials"][0]
    assert trial["status"] == "completed"
    assert trial["review"] == {}


@pytest.mark.parametrize("failure,category", [
    (FakeSDK.APIStatusError("bad request"), "api_status"),
    (FakeSDK.APITimeoutError("timeout"), "api_timeout"),
    (FakeSDK.APIConnectionError("offline"), "api_connection"),
])
def test_sdk_failures_are_typed_and_not_scored(monkeypatch, failure, category):
    FakeSDK.failure = failure
    monkeypatch.setattr(runner, "_load_session", lambda path: {"session_id": "s"})
    monkeypatch.setattr(runner, "build_fixture", lambda data: ({"evidence": []}, "f", "b"))
    trial = runner.run_pilot(_args(calls=1))["trials"][0]
    assert trial["status"] == "transport_error"
    assert trial["error_category"] == category
    assert trial["scoring"] is None
    assert "bad request" not in json.dumps(trial)
