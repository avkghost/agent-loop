"""Example custom command: info."""

def handle_info(args):
    """Display information about available agents and skills."""
    from src.agents_loader import load_agents
    from src.skills_loader import load_skills

    print("\n📚 Agent Loop Information\n")
    print("This is an autonomous agent loop with support for:")
    print("  • Agents: Named personas with customized system prompts")
    print("  • Skills: Provider-agnostic capability packages")
    print("  • Commands: Custom CLI commands for extending functionality\n")

    agents = load_agents()
    skills = load_skills()

    print(f"📊 Statistics:")
    print(f"  Agents loaded: {len(agents)}")
    print(f"  Skills loaded: {len(skills)}\n")

    print("💡 Try these commands:")
    print("  python3 agent_loop.py agents      - List all available agents")
    print("  python3 agent_loop.py skills      - List all available skills")
    print("  python3 agent_loop.py agent code-reviewer '<goal>'")
    print("  python3 agent_loop.py skill data-analyst '<goal>'\n")


COMMAND = {
    "name": "info",
    "description": "Display information about agents, skills, and available commands",
    "handler": handle_info
}
