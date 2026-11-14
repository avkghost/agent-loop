"""B3.6 dispatch guarantees: cancellation of a running tool call, verified through the real loop path."""

import asyncio
import json

from src import loop, runs, session
from src.runtime import AgentRuntime
from tests import scripted_provider
from tests.conftest import make_streamed_response


def test_cancelling_a_running_shell_call_is_recorded_as_interrupted_not_completed(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(loop, "load_project_guidelines", lambda: None)
    monkeypatch.setattr(session, "SESSIONS_DIR", tmp_path)
    slow = 'python3 -c "import time; time.sleep(20)"'
    scripted = [make_streamed_response(finish_reason="tool_calls", tool_calls=[
        {"id": "slow1", "name": "run_command", "arguments": json.dumps({"command": slow, "timeout_seconds": 60})}])]
    monkeypatch.setattr(scripted_provider, "POST", lambda *a, **k: scripted.pop(0))

    rt = AgentRuntime()
    try:
        sess = rt.open_session("cancel-shell")
        task = rt.event_loop.create_task(sess.run_goal("run something slow"))
        # Let the call start: the model turn is consumed, then the shell command is awaited
        for _ in range(400):
            rt.event_loop.run_until_complete(asyncio.sleep(0.01))
            record = next((r for r in (session.load_session("cancel-shell") or {}).get("execution", [])
                           if r.get("tool_call_id") == "slow1"), None)
            if record and record.get("status") == "running":
                break
        assert record and record["status"] == "running", "the shell call must be in flight before cancelling"

        task.cancel()
        try:
            rt.event_loop.run_until_complete(task)
        except asyncio.CancelledError:
            pass

        saved = session.load_session("cancel-shell")
        session.validate_session_data(saved)
        assert [r["status"] for r in saved["runs"]] == [runs.RunStatus.INTERRUPTED.value]
        call = next(r for r in saved["execution"] if r["tool_call_id"] == "slow1")
        assert call["status"] != "completed", "a cancelled call must never be recorded as completed"
        assert call["status"] in ("unknown", "failed", "running")
        sess.close()
    finally:
        rt.shutdown()
