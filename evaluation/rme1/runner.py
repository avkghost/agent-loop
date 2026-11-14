"""RME-1.1 agent-side runner. Runs inside the trial container as UID 10001. Contains NO hidden evaluator checks.

Prints bounded JSON lines: RESULT_JSON (summary) and SESSION_JSON (saved session document). The host supervisor
captures them and runs the independent evaluator separately.
"""

import json
import os
import sys
from pathlib import Path

# Point the existing provider at the approved endpoint BEFORE the provider module is imported.
from src import config  # noqa: E402
config.LM_STUDIO_URL = os.environ["RME_LLM_URL"]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scenarios  # noqa: E402
from src import loop, runs, session  # noqa: E402
from src.runtime import AgentRuntime  # noqa: E402

SESSION_LIMIT = 2 * 1024 * 1024


def main(scenario_id: str, workspace: str, session_dir: str) -> int:
    scenario = scenarios.SCENARIOS[scenario_id]
    os.chdir(workspace)
    config.ALLOWED_TOOLS = set(scenarios.TRIAL_TOOL_ALLOWLIST)
    config.MAX_ITERATIONS = scenarios.MAX_TURNS
    session.SESSIONS_DIR = Path(session_dir)
    sid = f"rme1-{scenario_id}"
    rt = AgentRuntime()
    status, reason, iterations = "runtime_error", "", 0
    try:
        sess = rt.open_session(sid)
        outcome = rt.event_loop.run_until_complete(sess.run_goal(
            scenario.goal, criteria=scenarios.build_criteria(scenario),
            verification_cycles=scenarios.VERIFICATION_CYCLES))
        sess.close()
        status, reason, iterations = outcome.run_status.value, outcome.reason, outcome.iteration_count
    except Exception as exc:  # recorded for the supervisor, never hidden
        reason = f"{type(exc).__name__}: {exc}"[:300]
    finally:
        rt.shutdown()
    saved = session.load_session(sid) or {}
    assistant_turns = sum(1 for m in saved.get("messages", []) if m.get("role") == "assistant")
    summary = {
        "scenario": scenario_id, "run_status": status, "reason": reason[:300],
        "iteration_count": iterations, "assistant_turns": assistant_turns,
        "tool_calls": [{"name": r["tool_name"], "status": r["status"], "error_category": r.get("error_category")}
                       for r in saved.get("execution", [])][:200],
        "verification_results": [{"criterion_id": r["criterion_id"], "attempt": r["attempt"],
                                  "budget": r.get("budget"), "status": r["status"],
                                  "observation": r["observation"][:200]}
                                 for r in saved.get("verification_results", [])][:100],
        "uid": os.getuid(), "llm_url": config.LM_STUDIO_URL,
    }
    print("RESULT_JSON " + json.dumps(summary, sort_keys=True))
    session_text = json.dumps(saved, sort_keys=True)
    if len(session_text) <= SESSION_LIMIT:
        print("SESSION_JSON " + session_text)
    else:
        print("SESSION_JSON_TRUNCATED " + str(len(session_text)))
    return 0 if status == "completed_verified" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
