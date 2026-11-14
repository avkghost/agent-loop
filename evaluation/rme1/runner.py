"""RME agent-side runner. Runs inside the trial container as UID 10001. Contains NO hidden evaluator checks.

Contract:
- The model is taken only from RME_MODEL. There is no default: a missing value stops the run before any request.
- Every provider request's model field is recorded (actual_request_models), so the supervisor can compare it with
  the intended model.
- Permission requests are answered with an explicit denial. stdin is never read.
- Results are published as framed, checksummed blocks (see frame()) that the supervisor validates. Nothing is read
  from terminal formatting.
"""

import asyncio
import hashlib
import json
import os
import signal
import sys
from pathlib import Path

MODEL = os.environ.get("RME_MODEL", "")
if not MODEL:
    sys.exit("RME_MODEL is required: no default model is used in a trial")
POLICY = os.environ.get("RME_POLICY", "")
if not POLICY:
    sys.exit("RME_POLICY is required: the trial policy is never implied")

# Point the existing provider at the approved endpoint and the intended model BEFORE the runtime modules load.
from src import config  # noqa: E402
config.LM_STUDIO_URL = os.environ["RME_LLM_URL"]
config.MODEL_NAME = MODEL

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scenarios  # noqa: E402
import policy  # noqa: E402
from src import loop, provider, session  # noqa: E402
from src.runtime import AgentRuntime  # noqa: E402
import metrics  # noqa: E402

TURN_RECORDS = []
metrics.wrap_complete(provider.LMStudioProvider, TURN_RECORDS)

MARK = "RME_FRAME"
SESSION_LIMIT = 2 * 1024 * 1024
REQUEST_MODELS = []
REQUEST_TOOL_HASHES = []


def install_request_recorder() -> None:
    """Observe (never change) the model field of each request the provider builds."""
    real_build_payload = provider.build_payload

    def recording(messages, tools, options):
        payload = real_build_payload(messages, tools, options)
        REQUEST_MODELS.append(payload.get("model"))
        REQUEST_TOOL_HASHES.append(policy.tool_schema_sha256(payload.get("tools")))
        return payload

    provider.build_payload = recording


def install_non_interactive_denial() -> None:
    """Answer every permission prompt with 'No' (option 3). The interactive reader is replaced so stdin is never
    read; denial is the existing user-denied path, so the run stops as the runtime defines for a denied action."""
    loop._get_single_key_input = lambda prompt, valid_keys: "3"


def frame(kind: str, nonce: str, obj) -> str:
    text = json.dumps(obj, sort_keys=True)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"\n{MARK} {kind} BEGIN {nonce}\n{text}\n{MARK} {kind} END {nonce} {digest}\n"


class DeadlineStop(Exception):
    """Raised in place of CancelledError, so the runner can tell an external deadline from other cancellation."""


def main(scenario_id: str, workspace: str, session_dir: str) -> int:
    nonce = os.environ.get("RME_NONCE", "")
    if not nonce:
        sys.exit("RME_NONCE is required to frame results")
    scenario = scenarios.SCENARIOS[scenario_id]
    os.chdir(workspace)
    config.ALLOWED_TOOLS = set(scenarios.TRIAL_TOOL_ALLOWLIST)
    config.MAX_ITERATIONS = scenarios.MAX_TURNS
    session.SESSIONS_DIR = Path(session_dir)
    install_request_recorder()
    install_non_interactive_denial()
    try:
        policy_meta = policy.install(POLICY)  # before any request; a refusal here sends nothing to the model
    except policy.PolicyError as exc:
        sys.exit(f"policy refused: {exc}")
    sid = f"rme1-{scenario_id}"
    rt = AgentRuntime()
    state = {"task": None, "deadline": False}

    def on_sigterm(signum, frame):
        # The supervisor's deadline. Cancel the goal through its own path: the runtime records INTERRUPTED,
        # the provider stream and any running command are cancelled, and the session is saved before we exit.
        state["deadline"] = True
        if state["task"] is not None:
            rt.event_loop.call_soon_threadsafe(state["task"].cancel)

    signal.signal(signal.SIGTERM, on_sigterm)
    status, reason, iterations = "runtime_error", "", 0
    try:
        sess = rt.open_session(sid)
        goal_task = rt.event_loop.create_task(sess.run_goal(
            scenario.goal, criteria=scenarios.build_criteria(scenario),
            verification_cycles=scenarios.VERIFICATION_CYCLES))
        state["task"] = goal_task
        if state["deadline"]:
            goal_task.cancel()  # the deadline arrived before the goal started
        outcome = rt.event_loop.run_until_complete(goal_task)
        sess.close()
        status, reason, iterations = outcome.run_status.value, outcome.reason, outcome.iteration_count
    except asyncio.CancelledError:
        # The goal was cancelled. The runtime has already recorded and saved INTERRUPTED; read the saved status
        # rather than inventing one.
        saved_runs = (session.load_session(sid) or {}).get("runs", [])
        status = saved_runs[-1]["status"] if saved_runs else "interrupted"
        reason = "external deadline: the goal was cancelled and its state saved" if state["deadline"] \
            else "the goal was cancelled"
    except Exception as exc:  # recorded for the supervisor, never hidden
        reason = f"{type(exc).__name__}: {exc}"[:300]
    finally:
        rt.shutdown()
    saved = session.load_session(sid) or {}
    summary = {
        "scenario": scenario_id, "intended_model": MODEL,
        "actual_request_models": sorted({str(m) for m in REQUEST_MODELS}),
        "request_count": len(REQUEST_MODELS),
        "run_status": status, "reason": reason[:300], "iteration_count": iterations,
        "deadline_signalled": state["deadline"],
        "turn_metrics": TURN_RECORDS[:200],
        "assistant_turns": sum(1 for m in saved.get("messages", []) if m.get("role") == "assistant"),
        "tool_calls": [{"name": r["tool_name"], "status": r["status"], "error_category": r.get("error_category")}
                       for r in saved.get("execution", [])][:200],
        "verification_results": [{"criterion_id": r["criterion_id"], "attempt": r["attempt"],
                                  "budget": r.get("budget"), "status": r["status"],
                                  "observation": r["observation"][:200]}
                                 for r in saved.get("verification_results", [])][:100],
        "uid": os.getuid(), "llm_url": config.LM_STUDIO_URL,
        "policy": dict(policy_meta, tool_schema_sha256=sorted(set(REQUEST_TOOL_HASHES))),
    }
    sys.stdout.write(frame("RESULT", nonce, summary))
    session_text = json.dumps(saved, sort_keys=True)
    if len(session_text) <= SESSION_LIMIT:
        sys.stdout.write(frame("SESSION", nonce, saved))
    else:
        sys.stdout.write(frame("SESSION_TRUNCATED", nonce, {"bytes": len(session_text)}))
    sys.stdout.flush()
    return 0 if status == "completed_verified" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
