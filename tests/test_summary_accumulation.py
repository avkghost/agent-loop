"""Repeated compaction must replace the previous summary, never accumulate summaries.

Compaction is driven directly with a mocked summarizer. Nothing calls a real model or tool.
"""

from tests import scripted_provider  # noqa: E402
import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from src import cli, compaction, config, loop, session
from tests.conftest import protocol_violations

OBJECTIVE = "Objective: fix the parser"
MARK = "\n\n[Context Summary]\n"


class SummarizerStub:
    """Returns summaries S1, S2, ... in order and records every prompt it was sent."""

    def __init__(self, texts=None, fail_from=None):
        self.texts = list(texts or [])
        self.fail_from = fail_from
        self.prompts = []

    def __call__(self, url, json=None, timeout=None, stream=False):
        self.prompts.append(json["messages"][1]["content"])
        index = len(self.prompts)
        if self.fail_from is not None and index >= self.fail_from:
            raise ConnectionError("summarizer unavailable")
        text = self.texts[index - 1] if index - 1 < len(self.texts) else f"S{index}"

        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}]}

        return Response()


def _round(messages, round_index):
    """One loop iteration's worth of history: two tool-call groups and an assistant note."""
    messages = list(messages)
    for k in range(2):
        call_id = f"c{round_index}_{k}"
        messages.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": "t", "arguments": "{}"}}]})
        messages.append({"role": "tool", "tool_call_id": call_id, "content": "r" * 40})
    messages.append({"role": "assistant", "content": "step"})
    return messages


def _journal_for(messages):
    """The durable records a runtime would hold for a history: one per declared call, completed when answered."""
    answers = {m.get("tool_call_id"): m.get("content") for m in messages if m.get("role") == "tool"}
    records = []
    for message in messages:
        for call in message.get("tool_calls", []) or []:
            call_id = call["id"]
            answered = call_id in answers
            records.append({"tool_call_id": call_id, "tool_name": call["function"]["name"], "arguments": {},
                            "status": "completed" if answered else "running",
                            "outcome": answers.get(call_id), "error_category": None})
    return records


def _compact_rounds(monkeypatch, rounds, stub=None, goal=OBJECTIVE):
    stub = stub or SummarizerStub()
    monkeypatch.setattr(scripted_provider, "POST", stub)
    state = compaction.CompactionState(goal)
    stub.state = state
    messages = [{"role": "system", "content": "SYS"}, {"role": "user", "content": goal}]
    history = []
    event_loop = asyncio.new_event_loop()
    try:
        for r in range(rounds):
            messages = _round(messages, r)
            messages = event_loop.run_until_complete(
                compaction.compact_messages(messages, state=state)
            )
            history.append(copy.deepcopy(messages))
    finally:
        event_loop.close()
    return history, stub


def test_objective_does_not_grow_across_three_compactions(monkeypatch):
    history, _ = _compact_rounds(monkeypatch, rounds=3)

    for step, messages in enumerate(history, start=1):
        content = messages[1]["content"]
        assert content.count("[Context Summary]") <= 1, f"after compaction {step}: {content.count('[Context Summary]')} summaries"
        assert content == OBJECTIVE + MARK + f"S{step}", f"after compaction {step}: {content!r}"


def test_each_compaction_replaces_the_previous_summary(monkeypatch):
    history, _ = _compact_rounds(monkeypatch, rounds=3)

    assert history[-1][1]["content"] == OBJECTIVE + MARK + "S3"
    assert "S1" not in history[-1][1]["content"] and "S2" not in history[-1][1]["content"]


def test_previous_summary_is_included_in_the_next_summarization_input(monkeypatch):
    _, stub = _compact_rounds(monkeypatch, rounds=3)

    assert "S1" not in stub.prompts[0]
    assert "S1" in stub.prompts[1], "the second summary must build on the first"
    assert "S2" in stub.prompts[2], "the third summary must build on the second"


def test_objective_text_is_preserved_verbatim(monkeypatch):
    history, _ = _compact_rounds(monkeypatch, rounds=4)

    for messages in history:
        assert messages[1]["content"].startswith(OBJECTIVE)
        assert messages[0] == {"role": "system", "content": "SYS"}


def test_tool_call_groups_stay_valid_after_each_compaction(monkeypatch):
    history, _ = _compact_rounds(monkeypatch, rounds=4)

    for step, messages in enumerate(history, start=1):
        assert not protocol_violations(messages), f"after compaction {step}: {protocol_violations(messages)}"


def test_summarization_failure_after_success_keeps_the_bounded_previous_summary(monkeypatch):
    history, _ = _compact_rounds(monkeypatch, rounds=3, stub=SummarizerStub(fail_from=2))

    assert history[0][1]["content"] == OBJECTIVE + MARK + "S1"
    for messages in history[1:]:
        content = messages[1]["content"]
        assert content.count("[Context Summary]") <= 1, "a failed summary must never stack a second summary"
        assert content in (OBJECTIVE, OBJECTIVE + MARK + "S1"), f"unexpected content after a failed summary: {content!r}"
        assert not protocol_violations(messages)


def test_fallback_drops_a_previous_summary_that_exceeds_the_current_budget(monkeypatch):
    monkeypatch.setattr(config, "MAX_TOKENS", 10**6)
    history, stub = _compact_rounds(monkeypatch, rounds=1, stub=SummarizerStub(texts=["x" * 2000], fail_from=None))
    assert len(history[0][1]["content"]) > len(OBJECTIVE) + 1000

    monkeypatch.setattr(config, "MAX_TOKENS", 100)  # the stored summary no longer fits the budget
    messages = _round(history[0], 9)
    event_loop = asyncio.new_event_loop()
    monkeypatch.setattr(scripted_provider, "POST", lambda *a, **k: (_ for _ in ()).throw(ConnectionError("down")))
    try:
        result = event_loop.run_until_complete(
            compaction.compact_messages(messages, state=stub.state)
        )
    finally:
        event_loop.close()

    assert result[1]["content"] == OBJECTIVE, "an oversized previous summary must not survive a fallback"


def test_context_size_stays_bounded_across_many_compactions(monkeypatch):
    history, _ = _compact_rounds(monkeypatch, rounds=10, stub=SummarizerStub(texts=["Z" * 30] * 10))

    sizes = [len(json.dumps(messages)) for messages in history]
    assert sizes[2:] == [sizes[2]] * len(sizes[2:]), f"context keeps growing: {sizes}"


def test_goal_containing_the_summary_marker_is_preserved_as_text():
    goal = "please keep [Context Summary] text literal"
    assert loop.check_goal(goal) == goal


def test_legacy_session_with_merged_summaries_is_refused_on_resume(tmp_sessions, monkeypatch, capsys):
    legacy = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": OBJECTIVE + "\n\n[Context Summary]\nold\n\n[Context Summary]\nolder"},
    ]
    session.save_session("legacy", "goal", legacy)
    raw = json.loads((tmp_sessions / "legacy.json").read_text())
    raw.pop("format", None)
    (tmp_sessions / "legacy.json").write_text(json.dumps(raw))
    calls = []
    monkeypatch.setattr(loop, "agent_loop", lambda *a, **k: calls.append(a))

    cli.main(["resume", "legacy"])

    out = capsys.readouterr().out
    assert "ambiguous" in out
    assert calls == [], "a legacy session with merged summaries must not be resumed"


def test_new_sessions_are_written_with_the_format_version(tmp_sessions):
    session.save_session("fresh", "goal", [{"role": "system", "content": "SYS"}])
    assert json.loads((tmp_sessions / "fresh.json").read_text())["format"] == session.SESSION_FORMAT


def test_legacy_session_without_merged_summaries_still_resumes(tmp_sessions, monkeypatch):
    session.save_session("plain", "goal", [{"role": "system", "content": "SYS"},
                                           {"role": "user", "content": OBJECTIVE}])
    raw = json.loads((tmp_sessions / "plain.json").read_text())
    raw.pop("format", None)
    (tmp_sessions / "plain.json").write_text(json.dumps(raw))
    calls = []
    monkeypatch.setattr(loop, "agent_loop", lambda *a, **k: calls.append(a))

    cli.main(["resume", "plain"])

    assert len(calls) == 1


def test_multi_compaction_session_round_trip_resumes_with_one_summary(tmp_sessions, monkeypatch):
    history, stub = _compact_rounds(monkeypatch, rounds=3)
    session.save_session("multi", "goal", history[-1], objective=stub.state.objective, summary=stub.state.summary,
                         execution=_journal_for(history[-1]))
    reloaded = session.load_session("multi")["messages"]

    assert reloaded == history[-1]
    assert reloaded[1]["content"] == OBJECTIVE + MARK + "S3"


def test_cannot_compact_stops_the_run_with_an_explicit_message(monkeypatch, capsys):
    from tests.test_hardening_regression import _drive
    from tests.conftest import make_streamed_response

    class FailingSummarizer:
        def __call__(self, body):
            raise ConnectionError("summarizer down")

    tool = lambda i: make_streamed_response(finish_reason="tool_calls", tool_calls=[
        {"id": f"call_{i}", "name": "list_directory", "arguments": json.dumps({"path": str(i)})}])
    responses = [tool(i) for i in range(6)] + [make_streamed_response(finish_reason="stop", content="done")]

    dispatched = []
    run = _drive(monkeypatch, responses, lambda *a, **k: dispatched.append(a) or "ok",
                 summarizer=lambda body: FailingSummarizer()(body), COMPACT_THRESHOLD=1.0, MAX_TOKENS=10)

    out = capsys.readouterr().out
    assert "Stopping" in out and "cannot continue" in out
    assert len(run.requests) < 7, "the run must stop instead of sending an over-budget request"


def test_run_preserves_a_goal_containing_a_marker_in_provider_messages(monkeypatch):
    from tests.conftest import make_streamed_response
    posts = []
    monkeypatch.setattr(loop, "load_project_guidelines", lambda: None)
    monkeypatch.setattr(scripted_provider, "POST", lambda *a, **k: posts.append(k["json"]) or make_streamed_response())
    event_loop = asyncio.new_event_loop()
    try:
        event_loop.run_until_complete(loop.agent_loop_async(
            "fix [Context Summary] handling", SimpleNamespace(get_openai_tools=lambda: []), event_loop))
    finally:
        event_loop.close()
    assert posts[0]["messages"][1]["content"] == "Objective: fix [Context Summary] handling"


def test_structured_state_does_not_parse_marker_text_from_model_summary(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "model text [Context Summary] literal"}, "finish_reason": "stop"}]}

    monkeypatch.setattr(scripted_provider, "POST", lambda *a, **k: Response())
    marker_objective = "Objective: user [Context Summary] [Recent User Message] text"
    state = compaction.CompactionState(objective=marker_objective)
    messages = _round([{"role": "system", "content": "SYS"},
                       {"role": "user", "content": marker_objective}], 0)
    event_loop = asyncio.new_event_loop()
    try:
        result = event_loop.run_until_complete(
            compaction.compact_messages(messages, state=state)
        )
    finally:
        event_loop.close()

    assert state.objective == marker_objective
    assert state.summary == "model text [Context Summary] literal"
    assert result[1]["content"].startswith(marker_objective)


def test_format_two_merged_summary_is_rejected_even_when_version_is_present(tmp_sessions, monkeypatch, capsys):
    session.save_session("legacy-format", "goal", [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": OBJECTIVE + "\n\n[Context Summary]\nold"},
    ])
    path = tmp_sessions / "legacy-format.json"
    raw = json.loads(path.read_text())
    raw["format"] = 2
    raw.pop("objective", None)
    raw.pop("summary", None)
    path.write_text(json.dumps(raw))
    calls = []
    monkeypatch.setattr(loop, "agent_loop", lambda *a, **k: calls.append((a, k)))

    cli.main(["resume", "legacy-format"])

    assert "ambiguous" in capsys.readouterr().out
    assert calls == []


def test_format_three_resume_passes_structured_state_to_runtime(tmp_sessions, monkeypatch):
    session.save_session(
        "structured", "goal", [{"role": "system", "content": "SYS"}],
        objective=OBJECTIVE, summary="current summary",
    )
    calls = []
    monkeypatch.setattr(loop, "agent_loop", lambda *a, **k: calls.append((a, k)))

    cli.main(["resume", "structured"])

    assert calls[0][1]["resume_objective"] == OBJECTIVE
    assert calls[0][1]["resume_summary"] == "current summary"


@pytest.mark.parametrize("mutate", [
    lambda data: data.pop("objective"),
    lambda data: data.pop("summary"),
    lambda data: data.update(summary=[]),
    lambda data: data.update(messages={}),
])
def test_malformed_format_three_session_is_rejected(tmp_sessions, monkeypatch, capsys, mutate):
    session.save_session("bad-structured", "goal", [{"role": "system", "content": "SYS"}])
    path = tmp_sessions / "bad-structured.json"
    raw = json.loads(path.read_text())
    mutate(raw)
    path.write_text(json.dumps(raw))
    monkeypatch.setattr(loop, "agent_loop", lambda *a, **k: pytest.fail("invalid session resumed"))

    cli.main(["resume", "bad-structured"])

    assert "Cannot resume session" in capsys.readouterr().out
