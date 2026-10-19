"""Integration tests for agents, skills, and commands."""

import pytest
from pathlib import Path
from src.agents_loader import load_agents, parse_agent_file, get_agent_description
from src.skills_loader import load_skills, parse_skill_file, parse_skill_directory
from src.commands_loader import load_commands
from src import dispatch


class TestAgentLoading:
    """Tests for agent loading and parsing."""

    def test_load_agents(self):
        """Test loading all agents from agents directory."""
        agents = load_agents()
        assert len(agents) >= 3, "Should load at least 3 example agents"
        assert "code-reviewer" in agents
        assert "researcher" in agents
        assert "local-assistant" in agents

    def test_agent_structure(self):
        """Test that agents have required fields."""
        agents = load_agents()
        code_reviewer = agents["code-reviewer"]

        assert code_reviewer.name == "code-reviewer"
        assert code_reviewer.model is not None
        assert code_reviewer.temperature is not None
        assert code_reviewer.max_tokens > 0
        assert isinstance(code_reviewer.tools, list)
        assert len(code_reviewer.tools) > 0
        assert code_reviewer.system_prompt is not None

    def test_agent_tools(self):
        """Test that agents reference valid tools."""
        agents = load_agents()

        # code-reviewer should have read_file and list_directory
        assert "read_file" in agents["code-reviewer"].tools
        assert "list_directory" in agents["code-reviewer"].tools

        # local-assistant should have write_file
        assert "write_file" in agents["local-assistant"].tools

    def test_get_agent_description(self):
        """Test agent description extraction."""
        agents = load_agents()
        code_reviewer = agents["code-reviewer"]
        description = get_agent_description(code_reviewer)

        assert len(description) > 0
        assert "code reviewer" in description.lower()


class TestSkillLoading:
    """Tests for skill loading and parsing."""

    def test_load_skills(self):
        """Test loading all skills from skills directory."""
        skills = load_skills()
        assert len(skills) >= 3, "Should load at least 3 example skills"
        assert "data-analyst" in skills
        assert "research-assistant" in skills
        assert "python-helper" in skills

    def test_flat_skill_structure(self):
        """Test flat-form skill fields."""
        skills = load_skills()
        data_analyst = skills["data-analyst"]

        assert data_analyst.name == "data-analyst"
        assert data_analyst.description is not None
        assert len(data_analyst.description) > 0
        assert isinstance(data_analyst.allowed_tools, list)
        assert data_analyst.system_prompt is not None
        # Flat-form skills should not have resources
        assert data_analyst.scripts is None
        assert data_analyst.references is None
        assert data_analyst.assets is None

    def test_directory_skill_structure(self):
        """Test directory-form skill with resources."""
        skills = load_skills()
        python_helper = skills["python-helper"]

        assert python_helper.name == "python-helper"
        assert python_helper.description is not None
        # Directory-form skills should have resources
        assert python_helper.scripts is not None
        assert python_helper.references is not None
        assert python_helper.assets is not None

        # Check that resources are populated
        assert len(python_helper.scripts) > 0, "Should have scripts"
        assert len(python_helper.references) > 0, "Should have references"
        assert len(python_helper.assets) > 0, "Should have assets"

    def test_skill_resources_absolute_paths(self):
        """Test that skill resources have absolute paths."""
        skills = load_skills()
        python_helper = skills["python-helper"]

        for script_name, script_path in python_helper.scripts.items():
            assert Path(script_path).is_absolute(), f"Script path should be absolute: {script_path}"
            assert Path(script_path).exists(), f"Script should exist: {script_path}"

    def test_skill_allowed_tools(self):
        """Test skill allowed-tools restrictions."""
        skills = load_skills()

        data_analyst = skills["data-analyst"]
        assert "read_file" in data_analyst.allowed_tools
        assert "run_command" in data_analyst.allowed_tools
        assert "write_file" not in data_analyst.allowed_tools

        python_helper = skills["python-helper"]
        assert "write_file" in python_helper.allowed_tools


class TestCommandLoading:
    """Tests for custom command loading."""

    def test_load_commands(self):
        """Test loading custom commands."""
        commands = load_commands()
        assert len(commands) >= 1, "Should load at least 1 example command"
        assert "info" in commands

    def test_command_structure(self):
        """Test command fields and handler."""
        commands = load_commands()
        info_cmd = commands["info"]

        assert info_cmd.name == "info"
        assert len(info_cmd.description) > 0
        assert callable(info_cmd.handler)


class TestAllowedToolsEnforcement:
    """Tests for allowed-tools permission enforcement."""

    def test_allowed_tool_permitted(self):
        """Test that allowed tools are permitted."""
        result = dispatch.dispatch_tool(
            "read_file",
            {"path": "src/cli.py"},
            allowed_tools=["read_file", "list_directory"]
        )
        # Should not have a permission error (may have other errors like file not found)
        assert not (result.get("errorCategory") == "permission")

    def test_disallowed_tool_rejected(self):
        """Test that disallowed tools are rejected."""
        result = dispatch.dispatch_tool(
            "write_file",
            {"path": "test.txt", "content": "test"},
            allowed_tools=["read_file", "list_directory"]
        )
        # Should have permission error
        assert result.get("isError") == True
        assert result.get("errorCategory") == "permission"
        assert "not permitted" in result.get("content", "")

    def test_empty_allowlist_means_unrestricted(self):
        """Test that empty allowlist means no restriction."""
        # Empty list = no restriction
        result = dispatch.dispatch_tool(
            "write_file",
            {"path": "test.txt", "content": "test"},
            allowed_tools=[]
        )
        # Should not have permission error
        assert result.get("errorCategory") != "permission"

    def test_none_allowlist_means_unrestricted(self):
        """Test that None allowlist means no restriction."""
        # None = no restriction
        result = dispatch.dispatch_tool(
            "write_file",
            {"path": "test.txt", "content": "test"},
            allowed_tools=None
        )
        # Should not have permission error
        assert result.get("errorCategory") != "permission"

    def test_allowlist_error_message_lists_allowed_tools(self):
        """Test that permission error includes list of allowed tools."""
        result = dispatch.dispatch_tool(
            "write_file",
            {"path": "test.txt", "content": "test"},
            allowed_tools=["read_file", "run_command"]
        )
        # Error message should list the allowed tools
        assert "read_file" in result.get("content", "")
        assert "run_command" in result.get("content", "")
