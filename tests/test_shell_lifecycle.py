"""B1 process-lifecycle safety: repeated cancellation, external reaping, group membership, retention.

Real-process tests run harmless child interpreters in a temporary directory. Every test is bounded,
and teardown kills and reaps anything it left behind, so no test leaves processes running.
"""

import asyncio
import gc
import os
import shutil
import subprocess
import signal
import sys
import time
import warnings

import pytest

from src import config, envelope
from src.tools import shell

PY = sys.executable

HELPERS = {
    # Leader that ignores SIGTERM and records its pid
    "ignore_term.py": ("import os, signal, time\n"
                       "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                       "open('pid', 'w').write(str(os.getpid()))\n"
                       "time.sleep(60)\n"),
    # Same-group grandchild that outlives the leader
    "grandchild.py": "import time\ntime.sleep(60)\n",
    "leader_spawns.py": ("import subprocess, sys\n"
                         "p = subprocess.Popen([sys.executable, 'grandchild.py'], stdin=subprocess.DEVNULL,\n"
                         "                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
                         "open('gc', 'w').write(str(p.pid))\n"
                         "print('leader done', flush=True)\n"),
    # Same-group child that ignores SIGTERM; the leader records the child's pid before exiting
    "stubborn.py": "import signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\ntime.sleep(60)\n",
    "stubborn_leader.py": ("import subprocess, sys\n"
                           "p = subprocess.Popen([sys.executable, 'stubborn.py'], stdin=subprocess.DEVNULL,\n"
                           "                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
                           "open('child', 'w').write(str(p.pid))\n"
                           "print('leader done', flush=True)\n"),
}


@pytest.fixture
def lifecycle_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "ALLOWED_TOOLS", {PY, "echo"})
    monkeypatch.setattr(config, "ALLOW_ALL", False)
    for name, body in HELPERS.items():
        (tmp_path / name).write_text(body)
    yield tmp_path
    for proc in list(shell._UNREAPED):  # teardown: no retained group survives a failed test
        try:
            os.killpg(proc.pid, signal.SIGKILL)  # the whole group, not only the leader
        except OSError:
            pass
        try:
            proc.kill()
            proc.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            pass
    shell._UNREAPED.clear()
    for pid_file in tmp_path.glob("*"):
        if pid_file.name in ("pid", "child", "gc"):
            try:
                os.kill(int(pid_file.read_text()), signal.SIGKILL)
            except (ValueError, OSError):
                pass


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _gone(pid: int, seconds: float = 3.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return not _alive(pid)


def _read_pid(tmp_path, name: str) -> int:
    return int((tmp_path / name).read_text())


def _execute(command: str, timeout: int = 20):
    return asyncio.run(shell.execute_command_async(command, timeout))


def _deny_signals(monkeypatch):
    """Gate group signalling: while 'on', every signal is refused as if permission were denied."""
    state = {"on": True}
    real = shell._signal_group

    def gated(pgid, sig):
        return False if state["on"] else real(pgid, sig)

    monkeypatch.setattr(shell, "_signal_group", gated)
    return state


def _fast_escalation(monkeypatch):
    monkeypatch.setattr(shell, "GRACE_SECONDS", 0.2)
    monkeypatch.setattr(shell, "KILL_CONFIRM_SECONDS", 0.2)


async def _wait_for_file(path, seconds: float = 10.0):
    end = time.monotonic() + seconds
    while not path.exists():
        if time.monotonic() >= end:
            raise AssertionError(f"{path.name} never appeared")
        await asyncio.sleep(0.02)


# ---------------------------------------------------------------------------
# F1: repeated cancellation during cleanup
# ---------------------------------------------------------------------------

@pytest.mark.real_process
def test_second_cancellation_during_cleanup_still_stops_the_leader(lifecycle_env, monkeypatch):
    monkeypatch.setattr(shell, "GRACE_SECONDS", 2.0)
    monkeypatch.setattr(shell, "KILL_CONFIRM_SECONDS", 3.0)

    async def scenario():
        task = asyncio.ensure_future(shell.execute_command_async(f"{PY} ignore_term.py", 120))
        await _wait_for_file(lifecycle_env / "pid")
        await asyncio.sleep(0.2)
        task.cancel()            # first cancellation: cleanup starts (SIGTERM, then grace)
        await asyncio.sleep(0.5)  # cleanup is inside the grace window now
        task.cancel()            # second cancellation must not abandon the cleanup
        with pytest.raises(asyncio.CancelledError):
            await task           # cancellation still propagates
        return _read_pid(lifecycle_env, "pid")

    started = time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pid = asyncio.run(scenario())
        gc.collect()
    elapsed = time.monotonic() - started

    assert not _alive(pid), "the SIGTERM-ignoring leader must be SIGKILLed and reaped despite the second cancellation"
    assert elapsed < 2.0 + 3.0 + 5.0, "cleanup must stay bounded"
    assert not [w for w in caught if issubclass(w.category, ResourceWarning)], "transports must be closed"
    assert shell._UNREAPED == []


# ---------------------------------------------------------------------------
# F2: external reaping (ECHILD) is identity loss, not exit
# ---------------------------------------------------------------------------

@pytest.mark.real_process
def test_external_reaping_is_identity_loss_and_nothing_is_signalled(lifecycle_env, monkeypatch):
    signals = []
    real = shell._signal_group

    def spy(pgid, sig):
        signals.append((pgid, sig))
        return real(pgid, sig)

    monkeypatch.setattr(shell, "_signal_group", spy)
    previous = signal.signal(signal.SIGCHLD, signal.SIG_IGN)  # the kernel reaps our children itself
    try:
        with pytest.raises(envelope.ToolExecutionError, match="identity was lost") as exc:
            _execute(f"{PY} -c \"import sys; sys.exit(3)\"")
    finally:
        signal.signal(signal.SIGCHLD, previous)

    assert signals == [], "no group signal may follow loss of identity"
    assert "exit code" not in str(exc.value), "no exit status may be claimed"
    assert "may still be running" in str(exc.value)


# ---------------------------------------------------------------------------
# F3: a leader is not reaped while its group still has live members
# ---------------------------------------------------------------------------

@pytest.mark.real_process
def test_leader_with_live_same_group_grandchild_is_not_reaped_until_the_group_is_empty(lifecycle_env, monkeypatch):
    _fast_escalation(monkeypatch)
    deny = _deny_signals(monkeypatch)

    result = _execute(f"{PY} leader_spawns.py")
    grandchild = _read_pid(lifecycle_env, "gc")

    assert "could not confirm" in result["content"], "an unconfirmed group must not be reported as empty"
    assert len(shell._UNREAPED) == 1
    assert shell._UNREAPED[0].returncode is None, "the leader must stay unreaped while the group is populated"

    shell.reap_unreaped()
    assert len(shell._UNREAPED) == 1, "reaping must wait for verified group emptiness, not pipe EOF"
    assert _alive(grandchild)

    deny["on"] = False
    for _ in range(100):
        shell.reap_unreaped()
        if not shell._UNREAPED:
            break
        time.sleep(0.05)
    assert shell._UNREAPED == []
    assert _gone(grandchild), "retried escalation must reach the surviving group member"


# ---------------------------------------------------------------------------
# F4: retained state is bounded, never discarded, and cleaned up
# ---------------------------------------------------------------------------

@pytest.mark.real_process
def test_repeated_unconfirmed_termination_is_bounded_and_nothing_is_discarded(lifecycle_env, monkeypatch):
    _fast_escalation(monkeypatch)
    monkeypatch.setattr(shell, "MAX_RETAINED_LEADERS", 2)
    deny = _deny_signals(monkeypatch)

    pids = []
    for _ in range(2):
        with pytest.raises(envelope.ToolExecutionError, match="could not be confirmed"):
            _execute(f"{PY} ignore_term.py", timeout=1)
        pids.append(_read_pid(lifecycle_env, "pid"))
        (lifecycle_env / "pid").unlink()

    assert len(shell._UNREAPED) == 2
    assert all(_alive(pid) for pid in pids), "unconfirmed leaders must not be discarded"

    refused = _execute(f"{PY} -c \"open('started', 'w').close()\"")
    assert refused["isError"] is True
    assert "could not be confirmed" in refused["content"]
    assert not (lifecycle_env / "started").exists(), "nothing may start past the retention cap"
    assert len(shell._UNREAPED) == 2

    deny["on"] = False
    for _ in range(100):
        shell.reap_unreaped()
        if not shell._UNREAPED:
            break
        time.sleep(0.05)
    assert shell._UNREAPED == []
    assert all(_gone(pid) for pid in pids), "later cleanup must escalate and reap every retained leader"


@pytest.mark.real_process
def test_runtime_shutdown_stops_retained_groups_whose_identity_is_held(lifecycle_env, monkeypatch):
    _fast_escalation(monkeypatch)
    deny = _deny_signals(monkeypatch)
    with pytest.raises(envelope.ToolExecutionError):
        _execute(f"{PY} ignore_term.py", timeout=1)
    pid = _read_pid(lifecycle_env, "pid")
    assert _alive(pid)

    deny["on"] = False
    remaining = shell.shutdown_retained(seconds=3)

    assert remaining == []
    assert shell._UNREAPED == []
    assert _gone(pid), "runtime shutdown must stop a retained group it can still identify"


# ---------------------------------------------------------------------------
# F5: process-group membership is required; its absence is never an empty group
# ---------------------------------------------------------------------------

@pytest.mark.real_process
def test_unavailable_membership_is_never_treated_as_an_empty_group(lifecycle_env, monkeypatch):
    _fast_escalation(monkeypatch)
    monkeypatch.setattr(shell, "_group_members", lambda pgid: None)

    result = _execute(f"{PY} stubborn_leader.py")
    child = _read_pid(lifecycle_env, "child")

    assert "could not confirm" in result["content"], "unknown membership must be reported, not claimed empty"
    assert _gone(child), "fail-closed escalation must still reach a group member that ignores SIGTERM"


@pytest.mark.real_process
def test_unavailable_membership_on_timeout_is_not_reported_as_confirmed(lifecycle_env, monkeypatch):
    _fast_escalation(monkeypatch)
    monkeypatch.setattr(shell, "_group_members", lambda pgid: None)

    with pytest.raises(envelope.ToolExecutionError) as exc:
        _execute(f"{PY} ignore_term.py", timeout=1)

    assert "could not be confirmed" in str(exc.value)
    assert "confirmed stopped" not in str(exc.value)


def test_missing_ps_refuses_before_anything_starts(lifecycle_env, monkeypatch):
    real_which = shutil.which
    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: None if name == "ps" else real_which(name, *a, **k))

    result = _execute(f"{PY} -c \"open('started', 'w').close()\"")

    assert result["isError"] is True
    assert "'ps'" in result["content"]
    assert not (lifecycle_env / "started").exists(), "no process may start without a membership check"


@pytest.mark.real_process
def test_observed_external_reap_is_identity_lost_and_stop_signals_nothing(lifecycle_env, monkeypatch):
    import subprocess

    proc = subprocess.Popen([PY, "-c", "pass"], start_new_session=True,
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOWAIT)  # wait for exit without reaping
    os.waitpid(proc.pid, 0)                                 # another party reaps it

    signals = []
    real = shell._signal_group
    monkeypatch.setattr(shell, "_signal_group", lambda pgid, sig: signals.append((pgid, sig)) or real(pgid, sig))

    assert shell._observe_leader(proc) == shell.IDENTITY_LOST
    assert asyncio.run(shell._stop_group(proc)) is False
    assert signals == [], "a PGID whose leader was reaped elsewhere must never be signalled"
    assert proc.returncode != 0, "no exit status may be inferred from a lost identity"
