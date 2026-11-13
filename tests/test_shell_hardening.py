"""B1 shell hardening: the foreground contract, process lifecycle, bounded output, runtime path.

Tests marked real_process run harmless child interpreters (this Python executable) in a
temporary directory. Each test records child PIDs in files, and teardown kills any survivors.
Everything else is pure logic or mocked spawns.
"""

import asyncio
import os
import shlex
import subprocess
import signal
import sys
import time
from pathlib import Path

import pytest

from src import config, envelope, loop, session
from src.tools import shell
from src.tools import __init__ as tool_defs  # noqa: F401  (schema module, imported for side-effect-free check)
from src.tools import get_local_tools
from tests.test_phase_a_final import run_agent

PY = sys.executable


def py(code: str) -> str:
    return f"{shlex.quote(PY)} -c {shlex.quote(code)}"


@pytest.fixture
def shell_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "ALLOWED_TOOLS", {PY, "echo"})
    monkeypatch.setattr(config, "ALLOW_ALL", False)
    yield tmp_path
    for pid_file in tmp_path.glob("*.pid"):  # teardown: no orphans, even if an assertion failed
        try:
            os.kill(int(pid_file.read_text()), signal.SIGKILL)
        except (ValueError, OSError):
            pass
    time.sleep(0.2)
    shell.reap_unreaped()


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_gone(pid: int, seconds: float = 3.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return not _alive(pid)


def _wait_file(path: Path, seconds: float = 10.0) -> None:
    deadline = time.monotonic() + seconds
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert path.exists(), f"{path.name} never appeared"


def _run(command: str, timeout=None):
    return asyncio.run(shell.execute_command_async(command, timeout))


# ---------------------------------------------------------------------------
# Contract (no processes)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("command", [
    "python server.py &",
    "sleep 100 &",
    "echo a & echo b",
    "echo a &",
    "sleep 1 & echo done",
    "echo 'a & b' & ",
])
def test_background_ampersand_is_rejected(command):
    result = shell.execute_command(command)
    assert result["isError"] is True
    assert result["errorCategory"] == "validation"
    assert "Unmanaged background execution is not supported" in result["content"]


@pytest.mark.parametrize("command", ["(sleep 1 &)", "echo `sleep 1 &`", "echo $(sleep 1 &)", "cat <<EOF"])
def test_unsupported_grammar_is_rejected_before_any_process(command):
    result = shell.execute_command(command)
    assert result["isError"] is True
    assert "not supported" in result["content"]


@pytest.mark.parametrize("command", [
    'echo "a & b"',
    "python3 -c 'print(\"&\")'",
    "echo a\\&b",
    'echo "a \\" & b"',
    "echo a && echo b",
    "echo a 2>&1",
    "echo a &> out.txt",
    "echo a >& out.txt",
])
def test_quoted_escaped_and_operator_ampersands_are_not_background(command):
    assert shell._syntax_rejection(command) is None


@pytest.mark.parametrize("command,expected", [
    ('echo "unterminated', "malformed"),
    ("echo a \\", "malformed"),
])
def test_malformed_input_is_reported_as_malformed(command, expected):
    result = shell._syntax_rejection(command)
    assert result is not None and expected in result["content"]
    assert result["errorCategory"] == "validation"


@pytest.mark.parametrize("command", [
    "nohup echo x",
    "setsid echo x",
    "echo ok; disown",
    "echo ok | screen",
    "echo ok && tmux new",
])
def test_detaching_commands_are_rejected(command):
    result = shell.execute_command(command)
    assert result["isError"] is True
    assert "detaches processes" in result["content"]




def test_job_control_commands_are_no_longer_automatically_allowed():
    assert not ({"kill", "jobs", "bg", "fg"} & set(config.ALLOWED_TOOLS))


def test_job_control_command_now_requires_approval(shell_env, monkeypatch):
    monkeypatch.setattr(config, "ALLOWED_TOOLS", {"echo"})
    result = shell.execute_command("kill 1234")
    assert result.get("requiresPermission") is True
    assert result["identifier"] == "kill"


@pytest.mark.parametrize("value", [0, 601, "5", True, 1.5])
def test_timeout_seconds_must_be_an_integer_in_range(value):
    result = shell.execute_command("echo x", value)
    assert result["isError"] is True
    assert result["errorCategory"] == "validation"


def test_unallowlisted_binary_still_requires_approval(shell_env, monkeypatch):
    monkeypatch.setattr(config, "ALLOWED_TOOLS", {"echo"})
    result = shell.execute_command("somebinary-not-listed arg")
    assert result.get("requiresPermission") is True


def test_schema_matches_runtime_contract():
    run = next(t for t in get_local_tools() if t["function"]["name"] == "run_command")
    description = run["function"]["description"]
    props = run["function"]["parameters"]["properties"]
    assert "15 seconds" not in description
    assert "Default 120" in description and "1 to 600" in description
    assert props["timeout_seconds"]["minimum"] == 1 and props["timeout_seconds"]["maximum"] == 600
    assert "Background execution is not supported" in description


def test_bounded_reader_keeps_memory_bounded_for_unlimited_output():
    class Endless:
        def __init__(self, total):
            self.remaining = total

        async def read(self, n):
            take = min(n, self.remaining)
            self.remaining -= take
            return b"y" * take

    head, tail, total = asyncio.run(shell._read_bounded(Endless(8 << 20)))
    assert total == 8 << 20
    assert len(head) + len(tail) <= 1 << 20, "capture must stay under 1 MiB per stream regardless of the constant"


# ---------------------------------------------------------------------------
# Process execution (real, harmless child interpreters)
# ---------------------------------------------------------------------------

@pytest.mark.real_process
def test_successful_command_returns_output(shell_env):
    result = _run(py("print('hello')"))
    assert result["isError"] is False
    assert result["content"].strip() == "hello"


@pytest.mark.real_process
def test_nonzero_exit_is_reported_with_its_code(shell_env):
    result = _run(py("import sys; print('partial'); sys.exit(3)"))
    assert result["isError"] is False
    assert "partial" in result["content"] and "[exit code 3]" in result["content"]


@pytest.mark.real_process
def test_command_cannot_consume_the_repl_input_that_follows(shell_env, monkeypatch):
    """Pending input on the process's stdin belongs to the next prompt, not to the command."""
    read_end, write_end = os.pipe()
    os.write(write_end, b"NEXT-PROMPT\n")
    os.close(write_end)
    saved_stdin = os.dup(0)
    os.dup2(read_end, 0)
    try:
        result = _run(py("import sys; print(repr(sys.stdin.read()))"))
        remaining = os.read(0, 100)
    finally:
        os.dup2(saved_stdin, 0)
        for fd in (saved_stdin, read_end):
            os.close(fd)
    assert result["content"].strip() == "''", "the command read the REPL's pending input"
    assert remaining == b"NEXT-PROMPT\n"


@pytest.mark.real_process
def test_stdout_and_stderr_are_kept_separate(shell_env):
    result = _run(py("import sys; print('out'); sys.stderr.write('err')"))
    content = result["content"]
    assert content.startswith("out")
    assert "--- stderr ---" in content and content.endswith("err")


@pytest.mark.real_process
def test_empty_output_has_an_explicit_message(shell_env):
    assert _run(py("pass"))["content"] == "[Command completed with no visual output]"


@pytest.mark.real_process
def test_non_utf8_output_is_replaced_not_fatal(shell_env):
    result = _run(py("import sys; sys.stdout.buffer.write(b'\\xff ok')"))
    assert result["isError"] is False and "ok" in result["content"]


@pytest.mark.real_process
def test_large_output_is_truncated_with_an_explicit_note(shell_env):
    result = _run(py("import sys; sys.stdout.write('x' * 3000000)"))
    content = result["content"]
    assert len(content) < 2 * shell.MODEL_STREAM_BYTES + 200
    assert "output truncated" in content and "3000000 bytes" in content


@pytest.mark.real_process
def test_timeout_stops_the_process_group_and_is_unknown_outcome(shell_env):
    pid_file = shell_env / "child.pid"
    code = f"import os,time; open({str(pid_file)!r},'w').write(str(os.getpid())); time.sleep(30)"
    with pytest.raises(envelope.ToolExecutionError, match="timed out after 1 seconds"):
        _run(py(code), 1)
    pid = int(pid_file.read_text())
    assert _wait_gone(pid), "timed-out child survived"


@pytest.mark.real_process
def test_sigterm_during_grace_period_is_a_clean_stop(shell_env, monkeypatch):
    monkeypatch.setattr(shell, "GRACE_SECONDS", 2.0)
    pid_file = shell_env / "graceful.pid"
    code = (f"import os,sys,signal,time; signal.signal(signal.SIGTERM, lambda *a: sys.exit(0));"
            f" open({str(pid_file)!r},'w').write(str(os.getpid())); time.sleep(30)")
    with pytest.raises(envelope.ToolExecutionError, match="process group was confirmed stopped"):
        _run(py(code), 1)
    assert _wait_gone(int(pid_file.read_text()))


@pytest.mark.real_process
def test_ignored_sigterm_is_forced_after_the_grace_period(shell_env, monkeypatch):
    monkeypatch.setattr(shell, "GRACE_SECONDS", 0.3)
    pid_file = shell_env / "stubborn.pid"
    code = (f"import os,signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN);"
            f" open({str(pid_file)!r},'w').write(str(os.getpid())); time.sleep(30)")
    started = time.monotonic()
    with pytest.raises(envelope.ToolExecutionError, match="process group was confirmed stopped"):
        _run(py(code), 1)
    assert time.monotonic() - started < 10
    assert _wait_gone(int(pid_file.read_text()))


@pytest.mark.real_process
def test_grandchild_holding_the_pipe_is_stopped_on_timeout(shell_env):
    pid_file = shell_env / "grandchild.pid"
    grandchild = f"import os,time; open({str(pid_file)!r},'w').write(str(os.getpid())); time.sleep(30)"
    leader = f"import subprocess,sys; subprocess.Popen([{PY!r},'-c',{grandchild!r}]); sys.exit(0)"
    with pytest.raises(envelope.ToolExecutionError):
        _run(py(leader), 2)
    assert _wait_gone(int(pid_file.read_text()))


@pytest.mark.real_process
def test_same_group_descendant_is_swept_after_the_leader_exits(shell_env):
    """A same-group descendant with no pipes is stopped while the zombie leader reserves the group."""
    pid_file = shell_env / "swept.pid"
    grandchild = f"import os,time; open({str(pid_file)!r},'w').write(str(os.getpid())); time.sleep(30)"
    leader = (f"import subprocess,sys,time; subprocess.Popen([{PY!r},'-c',{grandchild!r}],"
              f" stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); time.sleep(0.5); sys.exit(0)")
    result = _run(py(leader))
    assert result["isError"] is False
    assert "could not confirm" not in result["content"]
    _wait_file(pid_file)
    assert _wait_gone(int(pid_file.read_text())), "same-group descendant survived the command"


@pytest.mark.real_process
def test_detached_descendant_is_not_tracked_and_is_not_claimed_stopped(shell_env):
    """A descendant that starts its own session is outside the group. It is reported, not stopped."""
    pid_file = shell_env / "detached.pid"
    grandchild = f"import os,time; open({str(pid_file)!r},'w').write(str(os.getpid())); time.sleep(30)"
    leader = (f"import subprocess,sys,time; subprocess.Popen([{PY!r},'-c',{grandchild!r}],"
              f" stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True);"
              f" time.sleep(0.5); sys.exit(0)")
    result = _run(py(leader))
    assert result["isError"] is False
    _wait_file(pid_file)
    assert _alive(int(pid_file.read_text())), "a detached descendant must not be swept by group signals"


@pytest.mark.real_process
def test_detached_descendant_holding_the_pipe_times_out_without_claiming_it_stopped(shell_env):
    pid_file = shell_env / "detached-pipe.pid"
    grandchild = f"import os,time; open({str(pid_file)!r},'w').write(str(os.getpid())); time.sleep(30)"
    leader = (f"import subprocess,sys; subprocess.Popen([{PY!r},'-c',{grandchild!r}], start_new_session=True);"
              f" sys.exit(0)")
    with pytest.raises(envelope.ToolExecutionError) as caught:
        _run(py(leader), 2)
    message = str(caught.value)
    assert "process group was confirmed stopped" in message
    assert "not tracked" in message
    _wait_file(pid_file)
    assert _alive(int(pid_file.read_text())), "the stop must not claim a detached descendant was stopped"


@pytest.mark.real_process
def test_termination_during_cancellation_stops_same_group_children(shell_env):
    leader_pid = shell_env / "cancel-leader.pid"
    child_pid = shell_env / "cancel-child.pid"
    child = f"import os,time; open({str(child_pid)!r},'w').write(str(os.getpid())); time.sleep(30)"
    code = (f"import subprocess,os,time,sys; open({str(leader_pid)!r},'w').write(str(os.getpid()));"
            f" subprocess.Popen([{PY!r},'-c',{child!r}]); time.sleep(30)")

    async def scenario():
        task = asyncio.ensure_future(shell.execute_command_async(py(code)))
        await asyncio.get_running_loop().run_in_executor(None, _wait_file, child_pid)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert _wait_gone(int(leader_pid.read_text()))
    assert _wait_gone(int(child_pid.read_text())), "same-group child survived cancellation"


@pytest.mark.real_process
def test_group_gone_before_escalation_sends_no_sigkill(shell_env, monkeypatch):
    sent = []
    real_signal = shell._signal_group
    monkeypatch.setattr(shell, "_signal_group", lambda pgid, sig: sent.append(sig) or real_signal(pgid, sig))
    proc = subprocess.Popen([PY, "-c", "import time; time.sleep(30)"], start_new_session=True,
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert asyncio.run(shell._stop_group(proc)) is True
    assert sent == [signal.SIGTERM], "SIGKILL must not be sent once SIGTERM has emptied the group"
    assert proc.returncode is not None


@pytest.mark.real_process
def test_stale_pgid_is_never_signalled(shell_env, monkeypatch):
    """Once the leader has been reaped, its PGID may belong to another group. It is never signalled."""
    proc = subprocess.Popen([PY, "-c", "pass"], start_new_session=True,
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proc.wait()
    sent = []
    monkeypatch.setattr(shell, "_signal_group", lambda pgid, sig: sent.append((pgid, sig)) or True)
    assert asyncio.run(shell._stop_group(proc)) is False
    assert sent == []


@pytest.mark.real_process
def test_exited_but_unreaped_leader_is_reaped_by_the_stop(shell_env):
    proc = subprocess.Popen([PY, "-c", "pass"], start_new_session=True,
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.3)  # exited, still a zombie
    assert shell._group_members(proc.pid) == [], "a zombie leader is not a live group member"
    assert asyncio.run(shell._stop_group(proc)) is True
    assert proc.returncode is not None


@pytest.mark.real_process
def test_unconfirmed_stop_is_reported_not_claimed(shell_env, monkeypatch):
    """When the group cannot be signalled, the outcome is reported as unconfirmed, never as stopped."""
    monkeypatch.setattr(shell, "GRACE_SECONDS", 0.2)
    monkeypatch.setattr(shell, "KILL_CONFIRM_SECONDS", 0.2)
    monkeypatch.setattr(shell.os, "killpg", lambda *a: (_ for _ in ()).throw(PermissionError("denied")))
    pid_file = shell_env / "unsignalable.pid"
    code = f"import os,time; open({str(pid_file)!r},'w').write(str(os.getpid())); time.sleep(30)"
    with pytest.raises(envelope.ToolExecutionError, match="could not be confirmed"):
        _run(py(code), 1)


@pytest.mark.real_process
def test_stopping_an_already_reaped_process_signals_nothing(shell_env, monkeypatch):
    """An already-reaped leader is not signalled: its PGID may now name another group."""
    sent = []
    monkeypatch.setattr(shell, "_signal_group", lambda pgid, sig: sent.append(sig) or True)

    async def scenario():
        proc = await asyncio.create_subprocess_exec(PY, "-c", "pass", start_new_session=True)
        await proc.wait()
        return await shell._stop_group(proc)

    assert asyncio.run(scenario()) is False
    assert sent == []


@pytest.mark.real_process
def test_cancellation_stops_the_command_and_propagates(shell_env):
    pid_file = shell_env / "cancel.pid"
    code = f"import os,time; open({str(pid_file)!r},'w').write(str(os.getpid())); time.sleep(30)"

    async def scenario():
        task = asyncio.ensure_future(shell.execute_command_async(py(code)))
        await asyncio.get_running_loop().run_in_executor(None, _wait_file, pid_file)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert _wait_gone(int(pid_file.read_text()))


# ---------------------------------------------------------------------------
# Runtime path: agent loop, journal, resume
# ---------------------------------------------------------------------------

@pytest.mark.real_process
def test_agent_runtime_runs_a_foreground_command_and_journals_it(shell_env, monkeypatch, tmp_sessions):
    llm = run_agent(monkeypatch, [
        __import__("tests.conftest", fromlist=["x"]).make_streamed_response(finish_reason="tool_calls", tool_calls=[
            {"id": "call_run", "name": "run_command", "arguments": __import__("json").dumps({"command": py("print('runtime-ok')")})}]),
        __import__("tests.conftest", fromlist=["x"]).make_streamed_response(finish_reason="stop", content="done"),
    ], session_id="b1-run")
    tool_msg = next(m for m in llm.payloads[1]["messages"] if m.get("role") == "tool")
    assert "runtime-ok" in tool_msg["content"]
    saved = session.load_session("b1-run")
    assert saved["execution"][0]["status"] == "completed"


@pytest.mark.real_process
def test_timeout_then_resume_does_not_run_the_command_again(shell_env, monkeypatch, tmp_sessions):
    from tests.conftest import make_streamed_response
    import json

    counter = shell_env / "count.txt"
    code = f"import time; open({str(counter)!r}, 'a').write('x'); time.sleep(30)"
    run_agent(monkeypatch, [
        make_streamed_response(finish_reason="tool_calls", tool_calls=[
            {"id": "call_slow", "name": "run_command",
             "arguments": json.dumps({"command": py(code), "timeout_seconds": 1})}]),
        make_streamed_response(finish_reason="stop", content="stopped"),
    ], session_id="b1-timeout")
    saved = session.load_session("b1-timeout")
    assert saved["execution"][0]["status"] == "unknown"

    run_agent(monkeypatch, [make_streamed_response(finish_reason="stop", content="resumed")],
              session_id="b1-timeout", resume_messages=saved["messages"],
              resume_objective=saved["objective"], resume_summary=saved["summary"],
              resume_execution=saved["execution"])
    assert counter.read_text() == "x", "resume must not rerun a command that already ran"


def test_non_posix_systems_are_rejected_not_weakened(monkeypatch):
    monkeypatch.setattr(shell.os, "name", "nt")
    result = shell.execute_command("echo hi")
    assert result["isError"] is True
    assert "requires a POSIX system" in result["content"]


@pytest.mark.real_process
def test_running_command_does_not_block_the_event_loop(shell_env):
    ticks = []

    async def ticker():
        for _ in range(20):
            ticks.append(time.monotonic())
            await asyncio.sleep(0.05)

    async def scenario():
        command = asyncio.ensure_future(shell.execute_command_async(py("import time; time.sleep(0.8)")))
        await ticker()
        return await command

    result = asyncio.run(scenario())
    assert result["isError"] is False
    assert len(ticks) == 20, "other coroutines must keep running while the command runs"


@pytest.mark.real_process
def test_journal_stores_a_bounded_result_for_huge_output(shell_env, monkeypatch, tmp_sessions):
    from tests.conftest import make_streamed_response
    import json

    run_agent(monkeypatch, [
        make_streamed_response(finish_reason="tool_calls", tool_calls=[
            {"id": "call_big", "name": "run_command",
             "arguments": json.dumps({"command": py("import sys; sys.stdout.write('z' * 5000000)")})}]),
        make_streamed_response(finish_reason="stop", content="done"),
    ], session_id="b1-big")
    saved = session.load_session("b1-big")
    outcome = saved["execution"][0]["outcome"]
    assert saved["execution"][0]["status"] == "completed"
    assert len(outcome) < 2 * shell.MODEL_STREAM_BYTES + 500
    assert len(json.dumps(saved)) < 100_000, "session JSON must not carry raw unlimited output"


@pytest.mark.real_process
def test_every_pipe_transport_is_explicitly_closed_after_a_command(shell_env, monkeypatch):
    """Explicit close is deterministic; the garbage collector would only warn about a missing one."""
    opened = []
    real_open = shell._open_reader

    async def recording_open(pipe):
        reader, transport = await real_open(pipe)
        closes = []
        real_close = transport.close

        def close():
            closes.append(True)
            real_close()

        transport.close = close
        opened.append(closes)
        return reader, transport

    monkeypatch.setattr(shell, "_open_reader", recording_open)
    result = _run(py("import sys; sys.stdout.write('out'); sys.stderr.write('err')"))

    assert result["isError"] is False
    assert len(opened) == 2, "stdout and stderr each get one transport"
    assert all(closes for closes in opened), "each pipe transport must be explicitly closed"
