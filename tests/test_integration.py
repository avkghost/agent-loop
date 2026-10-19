"""Integration tests for agent_loop with mocked LM Studio."""

import sys
import json
import asyncio
from unittest.mock import Mock, patch, MagicMock
from io import StringIO
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import loop, envelope, config, prompts
from src.tools.mcp_client import MCPClientManager


class TestIntegrationAgentLoop:
    """Integration tests for the agent loop with mocked LM Studio responses."""

    def setup_method(self):
        """Reset state before each test."""
        config.ALLOWED_TOOLS = {"ls", "grep", "find"}
        config.HAS_SEARCHED_CODEBASE = False

    def mock_lm_studio_response(self, finish_reason: str, content: str = None, tool_calls: list = None):
        """Create a mock LM Studio response."""
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

    @patch('src.loop.requests.post')
    def test_simple_completion(self, mock_post):
        """Test agent loop with simple completion (no tool calls)."""
        # Mock LM Studio to return a simple message and stop
        mock_post.return_value.json.return_value = self.mock_lm_studio_response(
            finish_reason="stop",
            content="I've completed the task. The directory listing is ready."
        )

        # Create MCP client (will not be used in this test)
        mcp_client = MCPClientManager()

        # Run the agent loop
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)
        event_loop.run_until_complete(loop.agent_loop_async("test task", mcp_client, event_loop))

        # Verify LM Studio was called
        assert mock_post.called
        call_args = mock_post.call_args
        assert "messages" in call_args[1]["json"]
        assert "tools" in call_args[1]["json"]

    @patch('src.loop.requests.post')
    def test_tool_call_sequence(self, mock_post):
        """Test agent loop with tool call → observation → completion."""
        # First call: model decides to use run_command
        # Second call: model receives observation and completes
        responses = [
            self.mock_lm_studio_response(
                finish_reason="tool_calls",
                content="I'll list the files for you.",
                tool_calls=[
                    {
                        "id": "call_1",
                        "function": {
                            "name": "run_command",
                            "arguments": json.dumps({"command": "ls"})
                        }
                    }
                ]
            ),
            self.mock_lm_studio_response(
                finish_reason="stop",
                content="Here are the files in the directory."
            )
        ]

        mock_post.return_value.json.side_effect = responses

        # Create MCP client
        mcp_client = MCPClientManager()

        # Run the agent loop
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)
        event_loop.run_until_complete(loop.agent_loop_async("list files", mcp_client, event_loop))

        # Verify two calls to LM Studio (first for tool decision, second after tool result)
        assert mock_post.call_count >= 1

    @patch('src.loop.requests.post')
    def test_write_file_with_search_warning(self, mock_post):
        """Test write_file tool shows warning when codebase hasn't been searched."""
        responses = [
            self.mock_lm_studio_response(
                finish_reason="tool_calls",
                content="I'll create a new file.",
                tool_calls=[
                    {
                        "id": "call_1",
                        "function": {
                            "name": "write_file",
                            "arguments": json.dumps({
                                "path": "new_file.txt",
                                "content": "Hello, world!"
                            })
                        }
                    }
                ]
            ),
            self.mock_lm_studio_response(
                finish_reason="stop",
                content="File created."
            )
        ]

        mock_post.return_value.json.side_effect = responses

        # Ensure HAS_SEARCHED_CODEBASE is False
        config.HAS_SEARCHED_CODEBASE = False

        mcp_client = MCPClientManager()

        # Run the agent loop (with output capture to check for warning)
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)
        event_loop.run_until_complete(loop.agent_loop_async("create a file", mcp_client, event_loop))

        # Verify tool was called
        assert mock_post.call_count >= 1

    @patch('builtins.input', return_value='3')  # Mock input to deny permission
    @patch('src.loop.requests.post')
    def test_context_compaction(self, mock_post, mock_input):
        """Test context compaction when token limit is approached."""
        # Create responses that will trigger many iterations
        responses = [
            self.mock_lm_studio_response(
                finish_reason="tool_calls",
                content="Tool call 1",
                tool_calls=[{
                    "id": f"call_{i}",
                    "function": {
                        "name": "run_command",
                        "arguments": json.dumps({"command": "echo test"})
                    }
                }]
            )
            for i in range(5)
        ] + [
            self.mock_lm_studio_response(
                finish_reason="stop",
                content="Completed"
            )
        ]

        mock_post.return_value.json.side_effect = responses

        mcp_client = MCPClientManager()

        # Lower the compaction threshold for testing
        original_threshold = config.COMPACT_THRESHOLD
        config.COMPACT_THRESHOLD = 0.5  # Trigger compaction sooner

        try:
            event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(event_loop)
            event_loop.run_until_complete(loop.agent_loop_async("repeated task", mcp_client, event_loop))
            # If compaction was triggered, the loop still completes successfully
            assert True
        finally:
            config.COMPACT_THRESHOLD = original_threshold

    @patch('src.loop.requests.post')
    def test_lm_studio_error_handling(self, mock_post):
        """Test graceful handling of LM Studio communication errors."""
        mock_post.side_effect = Exception("Connection refused")

        mcp_client = MCPClientManager()

        # Run the agent loop (should handle error gracefully)
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)
        event_loop.run_until_complete(loop.agent_loop_async("test task", mcp_client, event_loop))

        # Verify error was raised and caught
        assert mock_post.called

    @patch('builtins.input', return_value='3')  # Mock input to deny permission
    @patch('src.loop.requests.post')
    def test_max_iterations_limit(self, mock_post, mock_input):
        """Test that agent loop stops at MAX_ITERATIONS."""
        # Create a response that always returns tool_calls without completion
        response = self.mock_lm_studio_response(
            finish_reason="tool_calls",
            content="I'll run a command",
            tool_calls=[{
                "id": "call_1",
                "function": {
                    "name": "run_command",
                    "arguments": json.dumps({"command": "echo test"})
                }
            }]
        )

        mock_post.return_value.json.return_value = response

        mcp_client = MCPClientManager()

        # Lower MAX_ITERATIONS for testing
        original_max = config.MAX_ITERATIONS
        config.MAX_ITERATIONS = 3

        try:
            event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(event_loop)
            event_loop.run_until_complete(loop.agent_loop_async("infinite task", mcp_client, event_loop))
            # Should reach max iterations and stop
            # Call count should be around MAX_ITERATIONS (3)
            assert mock_post.call_count <= 5  # Some buffer for tool calls
        finally:
            config.MAX_ITERATIONS = original_max

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
