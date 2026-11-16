"""CLI interface with argparse subcommands."""

import sys
import argparse
import asyncio
import uuid
from pathlib import Path

from . import acceptance, config, loop, jobs, terminal_input, verification
from .authorization import SessionAuthorization
from .runtime import AgentRuntime, SessionError, load_mcp_config
from .config_loader import load_config, apply_config_overrides
from .session import (SESSION_FORMAT, SessionLoadError, SessionValidationError, find_session_by_run_id,
                      list_sessions, load_session, validate_session_data, resume_history)
from .execution import ExecutionJournal
from .tools.mcp_client import MCPClientManager
from .agents_loader import load_agents, get_agent_description
from .skills_loader import load_skills
from .commands_loader import load_commands


def _add_verification_options(parser) -> None:
    parser.add_argument("--criteria", metavar="PATH",
                        help="JSON document of trusted acceptance criteria; completion is verified against them")
    parser.add_argument("--verification-cycles", type=int, default=2, metavar="N",
                        help=f"verification cycles, 1 to {verification.MAX_VERIFICATION_CYCLES} (default 2)")


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
    _add_verification_options(run_parser)

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
    _add_verification_options(resume_parser)

    # sessions: List saved sessions
    sessions_parser = subparsers.add_parser("sessions", help="List saved sessions")

    # inspect: Show persisted execution state without resuming or dispatching anything
    inspect_parser = subparsers.add_parser("inspect", help="Inspect a saved session without running it")
    inspect_parser.add_argument("session_id", help="Session ID to inspect")

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

    if args and args[0] not in ["run", "repl", "resume", "sessions", "inspect", "tools", "config", "agent", "agents", "skill", "skills"]:
        args = ["run"] + args

    return parser.parse_args(args)


def _report(outcome) -> None:
    """Human-readable final status of one goal. The RunOutcome itself is returned to callers."""
    if outcome is None:
        return
    print(f"\n🧾 Run {outcome.run_id}: {outcome.status} — {outcome.reason}")


def _valid_cycles(cycles: int) -> bool:
    if isinstance(cycles, int) and not isinstance(cycles, bool) and 1 <= cycles <= verification.MAX_VERIFICATION_CYCLES:
        return True
    print(f"❌ --verification-cycles must be an integer from 1 to {verification.MAX_VERIFICATION_CYCLES}")
    return False


def _criteria_from_path(path: str):
    """Read, parse and check a criteria document before any session or model request exists. None means it was
    refused, and the reasons have been printed."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except UnicodeDecodeError:
        print(f"❌ Criteria file {path} is not UTF-8 text; nothing was run.")
        return None
    except OSError as e:
        print(f"❌ Cannot read criteria file {path}: {e.strerror or e}; nothing was run.")
        return None
    try:
        criteria = acceptance.parse_criteria_document(text, max_criteria=verification.VerificationEngine().max_checks)
    except acceptance.CriterionDocumentError as e:
        print(f"❌ Invalid criteria file {path}; nothing was run:")
        for problem in e.problems:
            print(f"   - {problem}")
        return None
    authorization = SessionAuthorization.from_config()
    problems = []
    for criterion in criteria:
        reason = verification.declaration_problem(criterion, authorization)
        if reason:
            problems.append(f"{criterion.criterion_id}: {reason}")
    if problems:
        print(f"❌ Criteria file {path} declares checks that could never run; nothing was run:")
        for problem in problems:
            print(f"   - {problem}")
        return None
    return criteria


def _latest_results(session_id: str, run_id: str):
    """The latest attempt per criterion for one run, read back from the saved session. None if it cannot be read."""
    try:
        data = load_session(session_id)
    except SessionLoadError:
        return None
    if not data or "criteria" not in data or "verification_results" not in data:
        return {}
    try:
        criteria = acceptance.validate_criteria(data["criteria"])
        results = verification.validate_results(data["verification_results"], criteria)
    except ValueError:
        return None
    return verification.VerificationEngine.current(item for item in results if item.run_id == run_id)


def _verdict(outcome, criteria, latest):
    status = outcome.status
    if status == "completed_verified":
        return "VERIFIED", f"all {len(criteria)} trusted criteria passed"
    if status == "completed_unverified":
        return ("COMPLETED WITHOUT TRUSTED CRITERIA",
                "the model stopped and nothing was verified; pass --criteria to verify completion")
    if status == "needs_attention":
        if criteria and latest and not all(
                latest.get(c.criterion_id) and latest[c.criterion_id].status == verification.VerificationStatus.PASSED.value
                for c in criteria):
            return "FAILED VERIFICATION", "required criteria did not pass within the verification budget"
        return "NEEDS ATTENTION", outcome.reason
    if status in ("blocked", "unknown", "provider_failed"):
        return "AUTHORIZATION OR EXECUTION FAILURE", outcome.reason
    if status in ("interrupted", "limit_reached"):
        return "INTERRUPTED OR INCOMPLETE", outcome.reason
    return status.upper(), outcome.reason


def _unresolved_unknowns(session_id: str) -> int:
    """Tool outcomes this session records as unknown. Nothing resolves them: they are never retried."""
    try:
        data = load_session(session_id)
    except SessionLoadError:
        return 0
    if not data:
        return 0
    return sum(1 for item in data.get("execution", []) if isinstance(item, dict) and item.get("status") == "unknown")


def _report_run(outcome, session_id: str, criteria) -> None:
    """Final status of a run that had (or lacked) trusted criteria, with each criterion's latest evidence."""
    if outcome is None:
        return
    latest = _latest_results(session_id, outcome.run_id) if criteria else {}
    label, detail = _verdict(outcome, criteria or [], latest)
    print(f"\n🧾 Run {outcome.run_id}: {label} — {detail}")
    print(f"   session_id: {session_id}")
    print(f"   status {outcome.status}: {outcome.reason}")
    unknown = _unresolved_unknowns(session_id)
    if unknown:
        print(f"   ⚠️  {unknown} tool outcome(s) in this session are recorded as unknown and were not retried. "
              "This verdict does not resolve them.")
    if not criteria:
        return
    if latest is None:
        print("   evidence: the verification results could not be read back from the session")
        latest = {}
    for criterion in criteria:
        result = latest.get(criterion.criterion_id)
        if result is None:
            line = "not checked"
        else:
            line = f"{result.status} (attempt {result.attempt} of {result.budget}): {result.observation}"
        print(f"   · {criterion.criterion_id} [{criterion.criterion_type}]: {line}")


def _inspect_session(session_id: str) -> None:
    """Print persisted session and execution state without changing or resuming it."""
    requested_id = session_id
    if session_id.startswith("run_"):
        resolved = find_session_by_run_id(session_id)
        if resolved:
            session_id = resolved
            print(f"ℹ️ Resolved run_id {requested_id} to session_id {session_id}")
    try:
        data = load_session(session_id)
    except SessionLoadError as e:
        print(f"❌ Cannot load session ({e.category}): {e}")
        return
    if not data:
        print(f"❌ Session not found: {session_id}")
        return
    try:
        validate_session_data(data)
    except SessionValidationError as e:
        print(f"❌ Cannot inspect session: {e}")
        return

    records = [item for item in data.get("execution", []) if isinstance(item, dict)]
    print(f"📂 Session: {data.get('session_id', session_id)}")
    print(f"   goal: {data.get('goal', '')}")
    print(f"   format: {data.get('format', 'legacy')}")
    print(f"   execution records: {len(records)}")
    if not records:
        print("   no execution records")
        return
    for record in records:
        status = record.get("status", "unknown")
        category = record.get("error_category") or "-"
        origin = record.get("origin_run_id") or "legacy"
        print(f"   • {record.get('tool_call_id', '<missing>')} {record.get('tool_name', '<missing>')} "
              f"status={status} category={category} origin={origin}")
        if status == "unknown":
            print(f"     outcome: {record.get('outcome')}")


def _close_repl(runtime, repl_session) -> None:
    """Close the REPL session and shut the runtime down. A refusal or failure is reported, not raised."""
    try:
        if repl_session is not None:
            repl_session.close()
        runtime.shutdown()
    except SessionError as e:
        print(f"❌ {e}")


def main(args=None):
    """Main CLI entry point. Managed jobs still running when the runtime exits are terminated."""
    try:
        return _main(args)
    finally:
        jobs.shutdown_runtime()


def _main(args=None):
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
        if not _valid_cycles(parsed.verification_cycles):
            return
        criteria = _criteria_from_path(parsed.criteria) if parsed.criteria is not None else []
        if criteria is None:
            return
        session_id = str(uuid.uuid4())
        outcome = loop.agent_loop(goal, session_id=session_id, criteria=criteria,
                                  verification_cycles=parsed.verification_cycles)
        _report_run(outcome, session_id, criteria)

    elif command == "repl":
        # One stable session for every goal in this REPL; it closes (and stops its jobs) on exit
        print("🔄 Starting interactive agent loop (type 'exit' or 'quit' to exit)")
        runtime = AgentRuntime(mcp_config=load_mcp_config())
        asyncio.set_event_loop(runtime.event_loop)
        repl_session = None
        try:
            repl_session = runtime.open_session()
            print(f"🧭 Session: {repl_session.session_id}")
            while True:
                try:
                    goal = terminal_input.read_goal("\n🎯 Enter goal (or 'exit'/'quit' to exit): ").strip()
                    if goal.lower() in ("exit", "quit"):
                        break
                    if not goal:
                        continue

                    task = runtime.event_loop.create_task(repl_session.run_goal(goal))
                    try:
                        outcome = runtime.event_loop.run_until_complete(task)
                    except KeyboardInterrupt:
                        # A goal interrupted inside its own run has already recorded itself and cleared its
                        # busy flag. Only a goal still pending is cancelled through its own path. A finished
                        # task must not be awaited again: the loop would wait forever on it.
                        if not task.done():
                            task.cancel()
                            try:
                                runtime.event_loop.run_until_complete(task)
                            except asyncio.CancelledError:
                                pass
                        raise
                    _report(outcome)
                except KeyboardInterrupt:
                    print("\n\nExiting...")
                    break
                except SessionError as e:
                    print(f"❌ {e}")
                    break
        finally:
            _close_repl(runtime, repl_session)

    elif command == "resume":
        try:
            session_data = load_session(parsed.session_id)
        except SessionLoadError as e:
            print(f"❌ Cannot load session ({e.category}): {e}")
            return
        if not session_data:
            print(f"❌ Session not found: {parsed.session_id}")
            return

        messages = session_data.get("messages", [])
        try:
            session_kind = validate_session_data(session_data)
        except SessionValidationError as e:
            print(f"❌ Cannot resume session: {e}")
            return

        prior_runs, prior_evidence, legacy_history = resume_history(session_data)
        resume_kwargs = {"resume_runs": prior_runs, "resume_evidence": prior_evidence,
                         "legacy_history": legacy_history}
        if session_kind in ("structured", "structured_legacy"):
            objective = session_data.get("objective")
            summary = session_data.get("summary")
            execution = session_data.get("execution", [])
            # Merge, never replace: the run history and legacy flag above must still reach the loop
            resume_kwargs.update({
                "resume_objective": objective,
                "resume_summary": summary,
                "resume_execution": execution,
                "resume_criteria": session_data.get("criteria", []),
                "resume_verification_results": session_data.get("verification_results", []),
            })
        else:
            legacy_objective = next(
                (m.get("content") for m in messages if m.get("role") == "user"),
                None,
            )
            if isinstance(legacy_objective, str):
                resume_kwargs["resume_objective"] = legacy_objective

        if not _valid_cycles(parsed.verification_cycles):
            return
        # The trusted contract persisted with the session. A resumed run is checked against it, never against a
        # silently empty set: supplying different criteria is refused, and dropping them is not possible.
        persisted = []
        if session_data.get("criteria"):
            persisted = acceptance.contract_of(acceptance.validate_criteria(session_data["criteria"]))
            # Restored checks are judged by the current configuration, not the one that was in force when they were saved
            current = SessionAuthorization.from_config()
            stale = [f"{item.criterion_id}: {reason}" for item in persisted
                     for reason in [verification.declaration_problem(item, current)] if reason]
            if stale:
                print("❌ The session's trusted criteria are not valid under the current configuration; nothing was run:")
                for problem in stale:
                    print(f"   - {problem}")
                return
        if parsed.criteria is not None:
            supplied = _criteria_from_path(parsed.criteria)
            if supplied is None:
                return
            if not persisted:
                print("❌ This session was started without trusted criteria, so criteria cannot be added on resume. "
                      "Start a new run with --criteria instead; nothing was run.")
                return
            if not acceptance.same_contract(supplied, persisted):
                print(f"❌ The criteria in {parsed.criteria} differ from the session's trusted criteria "
                      f"(contract {acceptance.contract_fingerprint(persisted)}); nothing was run.")
                return
        run_criteria = []
        if persisted:
            existing = {item["criterion_id"] for item in session_data["criteria"]}
            generation = len({item["run_id"] for item in session_data["criteria"]})
            try:
                run_criteria = acceptance.resumed_copies(persisted, existing, generation)
            except acceptance.CriterionDocumentError as e:
                print(f"❌ {e}; nothing was run.")
                return
            print(f"🔒 Trusted criteria carried over from the session: {len(run_criteria)} check(s), "
                  f"contract {acceptance.contract_fingerprint(persisted)}")

        print(f"📂 Resuming session: {session_data.get('goal', 'Unknown goal')}")
        outcome = loop.agent_loop(
            session_data.get("goal", ""),
            session_id=parsed.session_id,
            resume_messages=messages,
            criteria=run_criteria,
            verification_cycles=parsed.verification_cycles,
            **resume_kwargs,
        )
        _report_run(outcome, parsed.session_id, run_criteria)

    elif command == "sessions":
        sessions = list_sessions()
        if not sessions:
            print("No saved sessions found")
            return

        print(f"\n{'Session ID':<40} {'Goal':<30} {'Messages':<8}")
        print("-" * 80)
        for session in sessions:
            print(f"{session['session_id']:<40} {session['goal']:<30} {session['message_count']:<8}")

    elif command == "inspect":
        _inspect_session(parsed.session_id)

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
        _report(loop.agent_loop(
            parsed.goal,
            system_prompt_override=agent.system_prompt,
            tool_allowlist_override=agent.tools
        ))

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
        _report(loop.agent_loop(
            parsed.goal,
            system_prompt_override=skill.system_prompt,
            tool_allowlist_override=skill.allowed_tools
        ))


if __name__ == "__main__":
    main()
