"""Integration tests for agent_loop with mocked LM Studio.

Model replies are streamed (SSE) because that is the only response shape the loop reads.
Tool dispatch is patched, so no shell command or file write runs.
"""

from tests import scripted_provider  # noqa: E402
import sys
import json
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import loop, envelope, config, prompts
from src.tools.mcp_client import MCPClientManager
from tests.conftest import protocol_violations


class FakeStream:
    """A non-streaming-compatible stand-in for requests.Response(stream=True), built from a reply dict."""

    def __init__(self, reply):
        choice = reply["choices"][0]
        message = choice.get("message", {})
        delta = {}
        if message.get("content"):
            delta["content"] = message["content"]
        if message.get("tool_calls"):
            delta["tool_calls"] = [dict(tc, index=i) for i, tc in enumerate(message["tool_calls"])]
        chunks = [
            {"choices": [{"delta": delta, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": choice["finish_reason"]}]},
        ]
        self._lines = [f"data: {json.dumps(c)}" for c in chunks] + ["data: [DONE]"]

    def raise_for_status(self):
        return None

    def iter_lines(self):
        return iter(self._lines)

    def close(self):
        return None


def scripted_post(replies, summary_fails=True):
    """requests.post replacement: model replies come from `replies`; summarizer calls (no tools) fail."""
    queue = list(replies)

    def post(url, json=None, timeout=None, stream=False):
        if json is not None and "tools" not in json:
            if summary_fails:
                raise ConnectionError("summarizer unavailable in tests")
            raise AssertionError("summarizer was not expected")
        if not queue:
            raise AssertionError("unexpected extra model request")
        return FakeStream(queue.pop(0))

    return post


def run_agent(goal):
    mcp_client = MCPClientManager()
    event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(event_loop)
    try:
        event_loop.run_until_complete(loop.agent_loop_async(goal, mcp_client, event_loop))
    finally:
        event_loop.close()


class TestIntegrationAgentLoop:
    """Integration tests for the agent loop with mocked LM Studio responses."""

    def setup_method(self):
        """Reset state before each test, remembering the globals so teardown can restore them."""
        self._saved = (config.ALLOWED_TOOLS,)
        config.ALLOWED_TOOLS = {"ls", "grep", "find"}

    def teardown_method(self):
        (config.ALLOWED_TOOLS,) = self._saved

    def mock_lm_studio_response(self, finish_reason: str, content: str = None, tool_calls: list = None):
        """Create a mock LM Studio response (chat-completions shape)."""
        message = {}
        if content:
            message["content"] = content
        if tool_calls:
            message["tool_calls"] = tool_calls

        return {
            "choices": [
                {
                    "finish_reason": finish_reason,
                    "message": message
                }
            ]
        }

    @patch('tests.scripted_provider.POST')
    def test_simple_completion(self, mock_post):
        """A reply without tool calls ends the run after one request."""
        mock_post.side_effect = scripted_post([
            self.mock_lm_studio_response(
                finish_reason="stop",
                content="I've completed the task. The directory listing is ready."
            )
        ])

        run_agent("test task")

        assert mock_post.call_count == 1
        payload = mock_post.call_args[1]["json"]
        assert "messages" in payload and "tools" in payload

    @patch('src.dispatch.dispatch_tool_async', new_callable=AsyncMock, return_value="ls output")
    @patch('tests.scripted_provider.POST')
    def test_tool_call_sequence(self, mock_post, mock_dispatch):
        """A tool call is dispatched once, its result is sent back, and the run completes."""
        mock_post.side_effect = scripted_post([
            self.mock_lm_studio_response(
                finish_reason="tool_calls",
                content="I'll list the files for you.",
                tool_calls=[{"id": "call_1", "function": {"name": "run_command",
                                                          "arguments": json.dumps({"command": "ls"})}}]
            ),
            self.mock_lm_studio_response(finish_reason="stop", content="Here are the files in the directory.")
        ])

        run_agent("list files")

        assert mock_dispatch.call_count == 1
        assert mock_post.call_count == 2
        second = mock_post.call_args_list[1][1]["json"]["messages"]
        results = [m for m in second if m.get("role") == "tool"]
        assert [m["tool_call_id"] for m in results] == ["call_1"]
        assert not protocol_violations(second)

    @patch('src.dispatch.dispatch_tool_async', new_callable=AsyncMock, return_value="file written")
    @patch('tests.scripted_provider.POST')
    def test_write_file_with_search_warning(self, mock_post, mock_dispatch):
        """A write_file call is dispatched with its arguments, then the run completes."""
        mock_post.side_effect = scripted_post([
            self.mock_lm_studio_response(
                finish_reason="tool_calls",
                content="I'll create a new file.",
                tool_calls=[{"id": "call_1", "function": {"name": "write_file", "arguments": json.dumps({
                    "path": "new_file.txt", "content": "Hello, world!"})}}]
            ),
            self.mock_lm_studio_response(finish_reason="stop", content="File created.")
        ])

        run_agent("create a file")

        assert mock_dispatch.call_args[0][0] == "write_file"
        assert mock_dispatch.call_args[0][1] == {"path": "new_file.txt", "content": "Hello, world!"}
        assert mock_post.call_count == 2

    @patch('src.dispatch.dispatch_tool_async', new_callable=AsyncMock, side_effect=lambda *a, **k: "x" * 4000)
    @patch('tests.scripted_provider.POST')
    def test_context_compaction(self, mock_post, mock_dispatch):
        """Forced compaction: the summarizer is attempted, falls back to truncation, and every
        request stays protocol-valid until the run completes."""
        tool_turns = [
            self.mock_lm_studio_response(
                finish_reason="tool_calls",
                content=f"Tool call {i}",
                tool_calls=[{"id": f"call_{i}", "function": {"name": "run_command",
                                                             "arguments": json.dumps({"command": "echo test"})}}]
            )
            for i in range(5)
        ]
        mock_post.side_effect = scripted_post(
            tool_turns + [self.mock_lm_studio_response(finish_reason="stop", content="Completed")])

        original = (config.MAX_TOKENS, config.COMPACT_THRESHOLD)
        system_tokens = len(prompts.SYSTEM_PROMPT) // 4
        config.MAX_TOKENS = system_tokens + 2000  # forces compaction once several 1000-token results accumulate
        config.COMPACT_THRESHOLD = 1.0
        try:
            run_agent("repeated task")
        finally:
            config.MAX_TOKENS, config.COMPACT_THRESHOLD = original

        loop_requests = [c[1]["json"] for c in mock_post.call_args_list if "tools" in c[1]["json"]]
        summary_attempts = [c for c in mock_post.call_args_list if "tools" not in c[1]["json"]]
        assert len(summary_attempts) >= 1, "compaction never attempted a summary"
        assert len(loop_requests) == 6, "the run must reach the final completion"
        for index, body in enumerate(loop_requests):
            assert not protocol_violations(body["messages"]), f"request {index}: {protocol_violations(body['messages'])}"

    @patch('tests.scripted_provider.POST')
    def test_lm_studio_error_handling(self, mock_post):
        """A transport error ends the run without raising out of the loop."""
        mock_post.side_effect = Exception("Connection refused")

        run_agent("test task")

        assert mock_post.call_count == 1

    @patch('src.dispatch.dispatch_tool_async', new_callable=AsyncMock, return_value="ok")
    @patch('tests.scripted_provider.POST')
    def test_max_iterations_limit(self, mock_post, mock_dispatch):
        """The run makes exactly MAX_ITERATIONS requests when every reply asks for a tool."""
        always_tools = [self.mock_lm_studio_response(
            finish_reason="tool_calls",
            content="I'll run a command",
            tool_calls=[{"id": f"call_{i}", "function": {"name": "run_command",
                                                      "arguments": json.dumps({"command": "echo test"})}}]
        ) for i in range(10)]
        mock_post.side_effect = scripted_post(always_tools)

        original_max = config.MAX_ITERATIONS
        config.MAX_ITERATIONS = 3
        try:
            run_agent("infinite task")
        finally:
            config.MAX_ITERATIONS = original_max

        assert mock_post.call_count == 3
        assert mock_dispatch.call_count == 3

    def test_envelope_functions(self):
        """Test structured error envelope functions."""
        # Test success
        result = envelope.success("Operation successful")
        assert result["isError"] is False
        assert result["content"] == "Operation successful"

        # Test transient error
        result = envelope.error_transient("Timeout occurred")
        assert result["isError"] is True
        assert result["errorCategory"] == "transient"
        assert result["isRetryable"] is True

        # Test validation error
        result = envelope.error_validation("Bad input")
        assert result["isError"] is True
        assert result["errorCategory"] == "validation"

        # Test permission error
        result = envelope.error_permission("Access denied")
        assert result["isError"] is True
        assert result["errorCategory"] == "permission"
        assert result["isRetryable"] is False

    def test_system_prompt_content(self):
        """Test that system prompt contains expected guidance."""
        assert "PERCEIVE" in prompts.SYSTEM_PROMPT
        assert "REASON" in prompts.SYSTEM_PROMPT
        assert "ACT" in prompts.SYSTEM_PROMPT
        assert "OBSERVE" in prompts.SYSTEM_PROMPT
        assert "run_command" in prompts.SYSTEM_PROMPT
        assert "write_file" in prompts.SYSTEM_PROMPT

    def test_config_defaults(self):
        """Test that config has expected default values."""
        assert config.MODEL_NAME is not None
        assert config.MAX_ITERATIONS > 0
        assert config.MAX_TOKENS > 0
        assert 0 < config.COMPACT_THRESHOLD < 1
        assert isinstance(config.ALLOWED_TOOLS, set)


def run_all_tests():
    """Run all integration tests."""
    test_suite = TestIntegrationAgentLoop()

    tests = [
        ("Simple Completion", test_suite.test_simple_completion),
        ("Tool Call Sequence", test_suite.test_tool_call_sequence),
        ("Write File Warning", test_suite.test_write_file_with_search_warning),
        ("Context Compaction", test_suite.test_context_compaction),
        ("LM Studio Error", test_suite.test_lm_studio_error_handling),
        ("Max Iterations", test_suite.test_max_iterations_limit),
        ("Envelope Functions", test_suite.test_envelope_functions),
        ("System Prompt", test_suite.test_system_prompt_content),
        ("Config Defaults", test_suite.test_config_defaults),
    ]

    passed = 0
    failed = 0

    print("\n" + "=" * 70)
    print("INTEGRATION TEST SUITE - Agent Loop")
    print("=" * 70 + "\n")

    for test_name, test_func in tests:
        test_suite.setup_method()
        try:
            test_func()
            print(f"✅ {test_name}")
            passed += 1
        except Exception as e:
            print(f"❌ {test_name}")
            print(f"   Error: {str(e)[:100]}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed} tests")
    print("=" * 70 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
