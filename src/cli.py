"""CLI interface with argparse subcommands."""

import sys
import argparse
import asyncio
import uuid
from pathlib import Path

from . import config, loop
from .config_loader import load_config, apply_config_overrides
from .session import list_sessions, load_session
from .tools.mcp_client import MCPClientManager
from .agents_loader import load_agents, get_agent_description
from .skills_loader import load_skills
from .commands_loader import load_commands


def parse_args(args=None):
    """Parse command-line arguments with subcommands."""
    parser = argparse.ArgumentParser(
        description="Autonomous agent loop with MCP support",
        prog="agent-loop"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # run: Execute a single goal
    run_parser = subparsers.add_parser("run", help="Run agent for a goal")
    run_parser.add_argument("goal", help="Goal for the agent")
    run_parser.add_argument("--allow", action="store_true", help="Allow all tools")
    run_parser.add_argument("--config", help="Config file path")
    run_parser.add_argument("--cache", action="store_true", help="Enable response caching")

    # repl: Interactive REPL mode
    repl_parser = subparsers.add_parser("repl", help="Interactive REPL mode")
    repl_parser.add_argument("--allow", action="store_true", help="Allow all tools")
    repl_parser.add_argument("--config", help="Config file path")
    repl_parser.add_argument("--cache", action="store_true", help="Enable response caching")

    # resume: Resume a saved session
    resume_parser = subparsers.add_parser("resume", help="Resume a saved session")
    resume_parser.add_argument("session_id", help="Session ID to resume")
    resume_parser.add_argument("--allow", action="store_true", help="Allow all tools")
    resume_parser.add_argument("--cache", action="store_true", help="Enable response caching")

    # sessions: List saved sessions
    sessions_parser = subparsers.add_parser("sessions", help="List saved sessions")

    # tools: List available tools
    tools_parser = subparsers.add_parser("tools", help="List available tools")
    tools_parser.add_argument("--config", help="Config file path")

    # config: Show effective configuration
    config_parser = subparsers.add_parser("config", help="Show effective configuration")
    config_parser.add_argument("--config", help="Config file path")

    # agent: Run with a named agent
    agent_parser = subparsers.add_parser("agent", help="Run agent with a specific persona")
    agent_parser.add_argument("name", help="Agent name")
    agent_parser.add_argument("goal", help="Goal for the agent")
    agent_parser.add_argument("--allow", action="store_true", help="Allow all tools")
    agent_parser.add_argument("--config", help="Config file path")

    # agents: List available agents
    agents_parser = subparsers.add_parser("agents", help="List available agents")

    # skill: Run with a named skill
    skill_parser = subparsers.add_parser("skill", help="Run skill with a specific capability package")
    skill_parser.add_argument("name", help="Skill name")
    skill_parser.add_argument("goal", help="Goal for the skill")
    skill_parser.add_argument("--allow", action="store_true", help="Allow all tools")
    skill_parser.add_argument("--config", help="Config file path")

    # skills: List available skills
    skills_parser = subparsers.add_parser("skills", help="List available skills")

    # Backward compatibility: if first arg isn't a subcommand, prepend "run"
    if args is None:
        args = sys.argv[1:]

    if args and args[0] not in ["run", "repl", "resume", "sessions", "tools", "config", "agent", "agents", "skill", "skills"]:
        args = ["run"] + args

    return parser.parse_args(args)


def main(args=None):
    """Main CLI entry point."""
    parsed = parse_args(args)
    command = parsed.command

    # Default to repl if no command provided
    if not command:
        command = "repl"
        print("No command specified. Starting interactive REPL mode.")
        print("Use: python3 agent_loop.py run '<goal>' to run directly")
        print("     python3 agent_loop.py agent <name> '<goal>' to run with an agent\n")

    # Load config file if specified
    if hasattr(parsed, "config") and parsed.config:
        config_data = load_config(parsed.config)
        apply_config_overrides(config_data)

    # Set global flags
    if hasattr(parsed, "allow") and parsed.allow:
        config.ALLOW_ALL = True

    # Handle commands
    if command == "run":
        if not hasattr(parsed, "goal") or not parsed.goal:
            print("Error: 'run' command requires a goal argument")
            print("Usage: python3 agent_loop.py run '<goal>'")
            return
        goal = parsed.goal
        session_id = str(uuid.uuid4())
        loop.agent_loop(goal)

    elif command == "repl":
        # Interactive REPL with persistent MCP connections
        print("🔄 Starting interactive agent loop (type 'exit' or 'quit' to exit)")

        # Set up MCP connections once
        mcp_client = MCPClientManager()
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)

        try:
            import json
            from pathlib import Path
            config_path = Path("mcp_servers.json")
            if config_path.exists():
                with open(config_path) as f:
                    mcp_config = json.load(f)
            else:
                mcp_config = {}

            event_loop.run_until_complete(mcp_client.connect_all(mcp_config))

            # REPL loop
            while True:
                try:
                    goal = input("\n🎯 Enter goal (or 'exit'/'quit' to exit): ").strip()
                    if goal.lower() in ("exit", "quit"):
                        break
                    if not goal:
                        continue

                    session_id = str(uuid.uuid4())
                    loop.agent_loop(goal)
                except KeyboardInterrupt:
                    print("\n\nExiting...")
                    break
        finally:
            event_loop.run_until_complete(mcp_client.close_all())
            event_loop.close()

    elif command == "resume":
        session_data = load_session(parsed.session_id)
        if not session_data:
            print(f"❌ Session not found: {parsed.session_id}")
            return

        print(f"📂 Resuming session: {session_data.get('goal', 'Unknown goal')}")
        # TODO: Implement resume logic (reconstruct messages and continue loop)
        print("⚠️ Resume not yet implemented")

    elif command == "sessions":
        sessions = list_sessions()
        if not sessions:
            print("No saved sessions found")
            return

        print(f"\n{'Session ID':<40} {'Goal':<30} {'Messages':<8}")
        print("-" * 80)
        for session in sessions:
            print(f"{session['session_id']:<40} {session['goal']:<30} {session['message_count']:<8}")

    elif command == "tools":
        from .tools import get_all_tools

        mcp_client = MCPClientManager()
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)

        try:
            import json
            from pathlib import Path
            config_path = Path("mcp_servers.json")
            if config_path.exists():
                with open(config_path) as f:
                    mcp_config = json.load(f)
            else:
                mcp_config = {}

            event_loop.run_until_complete(mcp_client.connect_all(mcp_config))
            tools = get_all_tools(mcp_client)

            print(f"\n📋 Available Tools ({len(tools)} total):\n")
            for tool in tools:
                if tool.get("type") == "function":
                    func = tool.get("function", {})
                    print(f"  • {func.get('name')}")
                    print(f"    {func.get('description', 'No description').split(chr(10))[0]}\n")
        finally:
            event_loop.run_until_complete(mcp_client.close_all())
            event_loop.close()

    elif command == "config":
        config_data = load_config(parsed.config if hasattr(parsed, "config") else None)
        print("\n📋 Effective Configuration:\n")
        print(f"LM_STUDIO_URL: {config.LM_STUDIO_URL}")
        print(f"MODEL_NAME: {config.MODEL_NAME}")
        print(f"MAX_ITERATIONS: {config.MAX_ITERATIONS}")
        print(f"MAX_TOKENS: {config.MAX_TOKENS}")
        print(f"COMPACT_THRESHOLD: {config.COMPACT_THRESHOLD}")
        print(f"ALLOWED_TOOLS: {config.ALLOWED_TOOLS}")

        if config_data:
            print("\n📁 Loaded from config file:")
            print(f"  {config_data}")

    elif command == "agents":
        agents = load_agents()
        if not agents:
            print("No agents found")
            return

        print(f"\n🤖 Available Agents ({len(agents)} total):\n")
        for name, agent in agents.items():
            description = get_agent_description(agent)
            print(f"  • {name}")
            print(f"    {description}")
            print(f"    Model: {agent.model}, Tools: {len(agent.tools)}\n")

    elif command == "agent":
        agents = load_agents()
        if parsed.name not in agents:
            print(f"❌ Agent not found: {parsed.name}")
            return

        agent = agents[parsed.name]

        # Set global overrides
        if hasattr(parsed, "allow") and parsed.allow:
            config.ALLOW_ALL = True
        if hasattr(parsed, "config") and parsed.config:
            config_data = load_config(parsed.config)
            apply_config_overrides(config_data)

        # Run the agent with overrides
        loop.agent_loop(
            parsed.goal,
            system_prompt_override=agent.system_prompt,
            tool_allowlist_override=agent.tools
        )

    elif command == "skills":
        skills = load_skills()
        if not skills:
            print("No skills found")
            return

        print(f"\n💡 Available Skills ({len(skills)} total):\n")
        for name, skill in skills.items():
            form = "directory" if skill.scripts else "flat"
            print(f"  • {name} ({form})")
            print(f"    {skill.description}")
            if skill.allowed_tools:
                print(f"    Allowed tools: {', '.join(skill.allowed_tools)}\n")
            else:
                print(f"    Allowed tools: all\n")

    elif command == "skill":
        skills = load_skills()
        if parsed.name not in skills:
            print(f"❌ Skill not found: {parsed.name}")
            return

        skill = skills[parsed.name]

        # Set global overrides
        if hasattr(parsed, "allow") and parsed.allow:
            config.ALLOW_ALL = True
        if hasattr(parsed, "config") and parsed.config:
            config_data = load_config(parsed.config)
            apply_config_overrides(config_data)

        # Run the skill with overrides
        loop.agent_loop(
            parsed.goal,
            system_prompt_override=skill.system_prompt,
            tool_allowlist_override=skill.allowed_tools
        )


if __name__ == "__main__":
    main()
