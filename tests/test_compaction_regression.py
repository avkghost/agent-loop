import asyncio
import requests
import pytest

from src import compaction


def fallback_expected():
    """System prompt, objective and an explicitly labelled recent user message."""
    fixture = messages_fixture()
    return [fixture[0], {"role": "user", "content": "old request\n\n[Recent User Message]\nnew request"}] + fixture[6:]


def messages_fixture():
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old request"},
        {
            "role": "assistant",
            "content": "calling tools",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
                {"id": "call_2", "type": "function", "function": {"name": "b", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "one"},
        {"role": "tool", "tool_call_id": "call_2", "content": "two"},
        {"role": "user", "content": "new request"},
        {"role": "assistant", "content": "new answer"},
        {"role": "user", "content": "latest"},
    ]


def run_compaction(monkeypatch, response_json):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return response_json

    monkeypatch.setattr(compaction.requests, "post", lambda *args, **kwargs: Response())
    event_loop = asyncio.new_event_loop()
    try:
        return event_loop.run_until_complete(
            compaction.compact_messages(messages_fixture(), event_loop)
        )
    finally:
        event_loop.close()


def test_active_loop_calls_summarizing_helper(monkeypatch, run_loop, streamed_response):
    called = []
    original = compaction.compact_messages

    async def spy(*args, **kwargs):
        called.append(True)
        return await original(*args, **kwargs)

    monkeypatch.setattr(compaction, "compact_messages", spy)
    monkeypatch.setattr("src.loop.get_token_count", lambda text: 100)

    post, _ = run_loop(
        [streamed_response(finish_reason="stop", content="done")],
        MAX_TOKENS=1,
        COMPACT_THRESHOLD=0.5,
    )

    assert called == [True]
    assert len(post.payloads[0]["messages"]) == 2


def test_helper_success_preserves_summary_and_tail(monkeypatch):
    result = run_compaction(
        monkeypatch,
        {"choices": [{"message": {"content": "summary"}}]},
    )

    assert result[0]["role"] == "system"
    # The original objective and the summary share one user message; the tail follows
    assert result[1]["content"] == (
        "old request\n\n[Context Summary]\nsummary\n\n[Recent User Message]\nnew request"
    )
    assert result[2:] == messages_fixture()[6:]


def test_summary_includes_messages_before_adjusted_tail(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "summary"}}]}

    def post(*args, **kwargs):
        captured["payload"] = kwargs["json"]
        return Response()

    monkeypatch.setattr(compaction.requests, "post", post)
    event_loop = asyncio.new_event_loop()
    try:
        compaction_result = event_loop.run_until_complete(
            compaction.compact_messages(messages_fixture(), event_loop)
        )
    finally:
        event_loop.close()

    summary_prompt = captured["payload"]["messages"][1]["content"]
    assert '"tool_call_id": "call_2"' in summary_prompt
    assert compaction_result[2:] == messages_fixture()[6:]


def test_helper_exception_uses_hard_truncation(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("summarizer unavailable")

    monkeypatch.setattr(compaction.requests, "post", fail)
    event_loop = asyncio.new_event_loop()
    try:
        result = event_loop.run_until_complete(
            compaction.compact_messages(messages_fixture(), event_loop)
        )
    finally:
        event_loop.close()

    assert result == fallback_expected()


@pytest.mark.parametrize("response_json", [{"choices": []}, {"choices": [{"message": {}}]}])
def test_helper_empty_or_malformed_summary_uses_fallback(monkeypatch, response_json):
    result = run_compaction(monkeypatch, response_json)
    assert result == fallback_expected()


def test_protocol_valid_group_is_preserved_by_helper_fallback(monkeypatch):
    result = run_compaction(monkeypatch, {"choices": []})
    retained_ids = {
        message["tool_call_id"]
        for message in result
        if message.get("role") == "tool"
    }
    declared_ids = {
        tool_call["id"]
        for message in result
        for tool_call in message.get("tool_calls", [])
    }

    assert retained_ids == declared_ids == set()
    assert all(message.get("role") != "tool" for message in result)


def test_timeout_uses_protocol_valid_fallback(monkeypatch):
    def timeout(*args, **kwargs):
        raise requests.exceptions.Timeout("summary timeout")

    monkeypatch.setattr(compaction.requests, "post", timeout)
    event_loop = asyncio.new_event_loop()
    try:
        result = event_loop.run_until_complete(
            compaction.compact_messages(messages_fixture(), event_loop)
        )
    finally:
        event_loop.close()

    assert result == fallback_expected()


def test_oversized_summary_uses_fallback(monkeypatch):
    from src import config

    monkeypatch.setattr(config, "MAX_TOKENS", 20)
    monkeypatch.setattr(config, "COMPACT_THRESHOLD", 1.0)
    result = run_compaction(
        monkeypatch,
        {"choices": [{"message": {"content": "x" * 1000}}]},
    )
    assert result[1]["content"] == "old request"
    assert not any("[Context Summary]" in (m.get("content") or "") for m in result)


def _compact_messages(monkeypatch, messages, summaries, state=None):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": summaries.pop(0)}}]}

    prompts_seen = []

    def post(*args, **kwargs):
        prompts_seen.append(kwargs["json"]["messages"][1]["content"])
        return Response()

    monkeypatch.setattr(compaction.requests, "post", post)
    event_loop = asyncio.new_event_loop()
    try:
        result = event_loop.run_until_complete(
            compaction.compact_messages(messages, event_loop, state=state)
        )
    finally:
        event_loop.close()
    return result, prompts_seen


def test_repeated_compaction_replaces_summary_and_carries_previous_summary(monkeypatch):
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Objective: preserve this objective"},
        {"role": "assistant", "content": "old work"},
        {"role": "user", "content": "request one"},
        {"role": "assistant", "content": "answer one"},
        {"role": "user", "content": "request two"},
        {"role": "assistant", "content": "answer two"},
        {"role": "user", "content": "request three"},
        {"role": "assistant", "content": "answer three"},
    ]

    state = compaction.CompactionState(messages[1]["content"])
    first, first_prompt = _compact_messages(monkeypatch, messages, ["summary one"], state)
    second_input = first + [
        {"role": "user", "content": "request four"},
        {"role": "assistant", "content": "answer four"},
    ]
    second, second_prompt = _compact_messages(monkeypatch, second_input, ["summary two"], state)

    assert first[1]["content"].count("[Context Summary]") == 1
    assert second[1]["content"].count("[Context Summary]") == 1
    assert "summary one" in second_prompt[0]
    assert second[1]["content"].startswith("Objective: preserve this objective")
    assert "summary one" not in second[1]["content"]
    assert "summary two" in second[1]["content"]
    assert "Objective: preserve this objective" not in first_prompt[0]


def test_three_compactions_keep_summary_and_objective_bounded(monkeypatch):
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Objective: bounded objective"},
        {"role": "assistant", "content": "old"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
        {"role": "assistant", "content": "a3"},
    ]
    lengths = []
    for index in range(3):
        if index == 0:
            state = compaction.CompactionState(messages[1]["content"])
        messages, _ = _compact_messages(monkeypatch, messages, [f"summary {index}"], state)
        lengths.append(len(messages[1]["content"]))
        messages += [
            {"role": "user", "content": f"next {index}"},
            {"role": "assistant", "content": f"answer {index}"},
        ]

    assert lengths == sorted(lengths)
    assert lengths[-1] < 100
    assert all(message["content"].count("[Context Summary]") <= 1
               for message in messages if message.get("role") == "user")


def test_failed_compaction_after_success_drops_old_summary_without_dropping_objective(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "first summary"}}]}

    monkeypatch.setattr(compaction.requests, "post", lambda *a, **k: Response())
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Objective: durable objective"},
        {"role": "assistant", "content": "old"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
        {"role": "assistant", "content": "a3"},
    ]
    event_loop = asyncio.new_event_loop()
    try:
        state = compaction.CompactionState(messages[1]["content"])
        compacted = event_loop.run_until_complete(compaction.compact_messages(messages, event_loop, state=state))

        def fail(*args, **kwargs):
            raise requests.exceptions.Timeout("second summary failed")

        monkeypatch.setattr(compaction.requests, "post", fail)
        compacted += [{"role": "user", "content": "u4"}, {"role": "assistant", "content": "a4"}]
        fallback = event_loop.run_until_complete(compaction.compact_messages(compacted, event_loop, state=state))
    finally:
        event_loop.close()

    assert fallback[1]["content"].startswith("Objective: durable objective")
    assert "first summary" not in fallback[1]["content"]
    assert "[Context Summary]" not in fallback[1]["content"]


def test_legacy_accumulated_summary_is_normalized_on_next_compaction(monkeypatch):
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old request\n\n[Context Summary]\nlegacy one\n\nlatest request"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "new request"},
        {"role": "assistant", "content": "new answer"},
        {"role": "user", "content": "final request"},
    ]
    with pytest.raises(compaction.CannotCompactError, match="ambiguous"):
        _compact_messages(monkeypatch, messages, ["normalized"])


def test_compaction_reports_when_objective_alone_cannot_fit(monkeypatch):
    from src import config

    monkeypatch.setattr(config, "MAX_TOKENS", 1)
    with pytest.raises(compaction.CannotCompactError, match="Cannot compact conversation"):
        run_compaction(monkeypatch, {"choices": [{"message": {"content": "summary"}}]})
