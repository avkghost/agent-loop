"""git_status and git_diff: fixed, read-only Git inspection.

These tests create their own repositories in temporary directories, because the Docker build context excludes .git.
The Linux test image installs git, so the tests that need it run there; only environments without git skip them.
"""

import asyncio
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from src import dispatch
from src.authorization import SessionAuthorization
from src.tools import git_tools, local_registry
from src.tools.registry import GIT

REAL_GIT = shutil.which("git")
needs_git = pytest.mark.skipif(REAL_GIT is None, reason="git is not installed in this environment")

_IDENTITY = {
    "GIT_AUTHOR_NAME": "Test", "GIT_AUTHOR_EMAIL": "test@example.invalid",
    "GIT_COMMITTER_NAME": "Test", "GIT_COMMITTER_EMAIL": "test@example.invalid",
    "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1",
}


def _git(cwd, *args):
    env = dict(os.environ, **_IDENTITY)
    subprocess.run([REAL_GIT, *args], cwd=cwd, env=env, check=True, capture_output=True)


def _snapshot(directory: Path):
    """Every file under a directory, with its bytes: any write to the repository metadata shows up here."""
    return {str(p.relative_to(directory)): p.read_bytes() for p in sorted(directory.rglob("*")) if p.is_file()}


@pytest.fixture
def repo(tmp_path, monkeypatch):
    workdir = tmp_path / "work"
    workdir.mkdir()
    _git(workdir, "init", "-q", "-b", "main")
    (workdir / "app.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    _git(workdir, "add", "app.py")
    _git(workdir, "commit", "-q", "-m", "initial")
    monkeypatch.chdir(workdir)
    return workdir


@pytest.fixture
def fake_slow_git(tmp_path, monkeypatch):
    """A git stand-in: rev-parse runs the real git, anything else sleeps. The pid of the sleeper is recorded."""
    pid_file = tmp_path / "sleeper.pid"
    script = tmp_path / "slow-git"
    script.write_text(
        "#!/bin/sh\n"
        "case \"$*\" in\n"
        f"  *rev-parse*) exec '{REAL_GIT}' \"$@\" ;;\n"
        "esac\n"
        f"echo $$ > '{pid_file}'\n"
        "exec sleep 30\n",
        encoding="utf-8")
    script.chmod(0o755)
    monkeypatch.setattr(git_tools, "_git_executable", lambda: str(script))
    return pid_file


# --- status -----------------------------------------------------------------------------------

@needs_git
def test_status_of_a_clean_repository(repo):
    result = git_tools.git_status_safe()
    assert result["isError"] is False
    assert result["content"] == "Branch: main\nWorking tree clean."


@needs_git
def test_status_reports_modified_untracked_and_staged_files(repo):
    (repo / "app.py").write_text("def a():\n    return 2\n", encoding="utf-8")
    (repo / "new.txt").write_text("new\n", encoding="utf-8")
    (repo / "staged.txt").write_text("staged\n", encoding="utf-8")
    _git(repo, "add", "staged.txt")
    lines = git_tools.git_status_safe()["content"].splitlines()
    assert lines[0] == "Branch: main"
    assert " M app.py" in lines
    assert "?? new.txt" in lines
    assert "A  staged.txt" in lines


@needs_git
def test_status_refuses_a_subdirectory_of_a_repository(repo, monkeypatch):
    sub = repo / "sub"
    sub.mkdir()
    monkeypatch.chdir(sub)
    result = git_tools.git_status_safe()
    assert result["isError"] is True and result["errorCategory"] == "validation"
    assert "not its top level" in result["content"]


# --- diff -------------------------------------------------------------------------------------

@needs_git
def test_diff_shows_the_change_made_by_edit_file(repo):
    result = asyncio.run(dispatch.dispatch_tool_async(
        "edit_file", {"path": "app.py", "old_string": "return 1", "new_string": "return 2"},
        authorization=SessionAuthorization()))
    assert result["isError"] is False
    diff = asyncio.run(dispatch.dispatch_tool_async("git_diff", {}, authorization=SessionAuthorization()))
    assert diff["isError"] is False
    assert diff["content"].startswith("Comparing the working tree with HEAD")
    assert "-    return 1" in diff["content"] and "+    return 2" in diff["content"]


@needs_git
def test_diff_with_a_path_is_limited_to_that_path(repo):
    (repo / "other.py").write_text("x = 1\n", encoding="utf-8")
    _git(repo, "add", "other.py")
    _git(repo, "commit", "-q", "-m", "other")
    (repo / "app.py").write_text("changed app\n", encoding="utf-8")
    (repo / "other.py").write_text("changed other\n", encoding="utf-8")
    only_app = git_tools.git_diff_safe("app.py")["content"]
    assert "app.py" in only_app and "other.py" not in only_app


@needs_git
def test_diff_with_no_changes_says_so(repo):
    result = git_tools.git_diff_safe()
    assert result["isError"] is False and result["content"].endswith("No differences.")


@needs_git
def test_diff_before_the_first_commit_compares_with_the_index(tmp_path, monkeypatch):
    workdir = tmp_path / "fresh"
    workdir.mkdir()
    _git(workdir, "init", "-q", "-b", "main")
    (workdir / "a.txt").write_text("staged content\n", encoding="utf-8")
    _git(workdir, "add", "a.txt")
    (workdir / "a.txt").write_text("worktree content\n", encoding="utf-8")
    monkeypatch.chdir(workdir)
    result = git_tools.git_diff_safe()
    assert result["isError"] is False
    assert "the index (no commits yet" in result["content"]
    assert "-staged content" in result["content"] and "+worktree content" in result["content"]


@needs_git
def test_diff_path_outside_the_workspace_is_refused(repo, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "x.txt"
    outside.write_text("secret\n", encoding="utf-8")
    for path in ("../" + outside.name, str(outside)):
        result = git_tools.git_diff_safe(path)
        assert result["isError"] is True and result["errorCategory"] == "permission"


@needs_git
def test_symlink_escape_in_the_diff_path_is_refused(repo, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "y.py"
    outside.write_text("secret\n", encoding="utf-8")
    (repo / "link.py").symlink_to(outside)
    result = git_tools.git_diff_safe("link.py")
    assert result["isError"] is True and result["errorCategory"] == "permission"


# --- non-Git directories, missing git, arguments ---------------------------------------------

def test_a_non_git_directory_is_a_clean_error(tmp_path, monkeypatch):
    plain = tmp_path / "plain"
    plain.mkdir()
    monkeypatch.chdir(plain)
    for result in (git_tools.git_status_safe(), git_tools.git_diff_safe()):
        assert result["isError"] is True and result["errorCategory"] == "validation"
        assert "not a usable Git repository" in result["content"]


def test_missing_git_is_a_clean_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(git_tools, "_git_executable", lambda: None)
    for result in (git_tools.git_status_safe(), git_tools.git_diff_safe()):
        assert result["isError"] is True and "not installed" in result["content"]


@needs_git
def test_unexpected_arguments_are_refused_before_anything_runs(repo):
    for tool_name, tool_input in (("git_status", {"args": ["--output=x"]}),
                                  ("git_diff", {"path": "app.py", "output": "x"}),
                                  ("git_diff", {"path": "app.py", "args": ["--ext-diff"]})):
        result = git_tools.call(tool_name, tool_input)
        assert result["isError"] is True and "does not accept" in result["content"]


@needs_git
def test_a_path_that_looks_like_an_option_is_only_a_path(repo):
    result = git_tools.git_diff_safe("--output=leak.txt")
    assert result["isError"] is False
    assert not (repo / "leak.txt").exists()


def test_empty_or_non_string_diff_path_is_refused(repo):
    for bad in ("", 7):
        result = git_tools.call("git_diff", {"path": bad})
        assert result["isError"] is True


# --- read-only behaviour, bounds, helpers ----------------------------------------------------

@needs_git
def test_inspection_writes_nothing_to_the_repository(repo):
    before = _snapshot(repo / ".git")
    (repo / "app.py").write_text("def a():\n    return 1\n", encoding="utf-8")  # same bytes, newer mtime
    os.utime(repo / "app.py", (time.time() + 60, time.time() + 60))
    git_tools.git_status_safe()
    git_tools.git_diff_safe()
    after = _snapshot(repo / ".git")
    assert after == before
    assert not (repo / ".git" / "index.lock").exists()


@needs_git
def test_repository_config_cannot_run_external_helpers(repo):
    marker = repo.parent / "helper-ran"
    helper = repo.parent / "helper.sh"
    helper.write_text(f"#!/bin/sh\necho ran >> '{marker}'\nexit 0\n", encoding="utf-8")
    helper.chmod(0o755)
    (repo / ".gitattributes").write_text("*.py diff=ext\n*.txt diff=conv\n", encoding="utf-8")
    _git(repo, "add", ".gitattributes")
    _git(repo, "commit", "-q", "-m", "attrs")
    (repo / "app.py").write_text("def a():\n    return 3\n", encoding="utf-8")
    (repo / "note.txt").write_text("one\n", encoding="utf-8")
    _git(repo, "add", "note.txt")
    _git(repo, "commit", "-q", "-m", "note")
    (repo / "note.txt").write_text("two\n", encoding="utf-8")
    # Configure helpers after the test's own Git commands have run, so only the tools under test can trigger them
    _git(repo, "config", "diff.ext.command", str(helper))
    _git(repo, "config", "diff.conv.textconv", str(helper))
    _git(repo, "config", "core.fsmonitor", str(helper))

    assert git_tools.git_status_safe()["isError"] is False
    diff = git_tools.git_diff_safe()
    assert diff["isError"] is False and "+    return 3" in diff["content"]
    assert not marker.exists()


@needs_git
def test_output_is_bounded_and_the_truncation_is_reported(repo):
    (repo / "app.py").write_text("".join(f"line {i} " + "x" * 80 + "\n" for i in range(2000)), encoding="utf-8")
    result = git_tools.git_diff_safe()
    assert result["isError"] is False
    assert f"output truncated at {git_tools.GIT_MAX_OUTPUT_BYTES} bytes" in result["content"]
    assert len(result["content"]) < git_tools.GIT_MAX_OUTPUT_BYTES + 400


@needs_git
def test_status_listing_is_bounded_and_says_when_it_is_incomplete(repo):
    for i in range(1500):
        (repo / f"untracked_file_with_a_long_name_{i:05d}_{'z' * 30}.txt").write_text("", encoding="utf-8")
    result = git_tools.git_status_safe()
    assert result["isError"] is False
    assert "this listing is incomplete" in result["content"]
    assert len(result["content"]) < git_tools.GIT_MAX_OUTPUT_BYTES + 400


@needs_git
def test_a_slow_git_is_stopped_at_the_time_limit_and_its_process_is_killed(repo, fake_slow_git, monkeypatch):
    # The budget also covers the real `git rev-parse` calls that run first, so it must allow for process start-up under load
    monkeypatch.setattr(git_tools, "GIT_TIMEOUT_SECONDS", 2.0)
    started = time.monotonic()
    result = git_tools.git_diff_safe()
    elapsed = time.monotonic() - started
    assert result["isError"] is True and result["errorCategory"] == "transient"
    assert elapsed < 5
    pid = int(fake_slow_git.read_text().strip())
    deadline = time.monotonic() + 3
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        assert time.monotonic() < deadline, "the timed-out git process is still running"
        time.sleep(0.05)


async def _wait_for_file(path: Path, seconds: float = 10.0) -> None:
    deadline = time.monotonic() + seconds
    while not path.exists():
        assert time.monotonic() < deadline, f"{path.name} never appeared"
        await asyncio.sleep(0.01)


def _ps_state(pid: int) -> str:
    return subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()


def _alive(pid: int) -> bool:
    state = _ps_state(pid)
    return bool(state) and not state.startswith("Z")


def _assert_gone(pid: int, seconds: float = 3.0) -> None:
    """For processes that are not ours to reap (an orphan reparented to init). Polls OS state only."""
    deadline = time.monotonic() + seconds
    while _alive(pid):
        assert time.monotonic() < deadline, f"process {pid} is still running after cancellation"
        time.sleep(0.05)


@pytest.fixture
def fake_git_with_child(tmp_path, monkeypatch):
    """A git stand-in that starts a background child in its own group and then waits. Both pids are recorded; the
    leader's pid file is written last, so when it exists the child is already running."""
    leader_file, child_file = tmp_path / "leader.pid", tmp_path / "child.pid"
    script = tmp_path / "child-git"
    script.write_text(
        "#!/bin/sh\n"
        "case \"$*\" in\n"
        f"  *rev-parse*) exec '{REAL_GIT}' \"$@\" ;;\n"
        "esac\n"
        f"sleep 60 &\necho $! > '{child_file}'\necho $$ > '{leader_file}'\nwait\n",
        encoding="utf-8")
    script.chmod(0o755)
    monkeypatch.setattr(git_tools, "_git_executable", lambda: str(script))
    return leader_file, child_file


@needs_git
@pytest.mark.parametrize("tool_name", ["git_status", "git_diff"])
def test_cancelling_a_git_call_stops_and_reaps_its_group_before_the_cancellation_completes(
        repo, fake_git_with_child, tmp_path, monkeypatch, tool_name):
    leader_file, child_file = fake_git_with_child
    scratch_root = tmp_path / "scratch"
    scratch_root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch_root))
    # A long limit, so that only the cancellation can stop git within the bound asserted below
    monkeypatch.setattr(git_tools, "GIT_TIMEOUT_SECONDS", 60)

    async def cancel_once_git_is_running():
        task = asyncio.create_task(dispatch.dispatch_tool_async(tool_name, {}, authorization=SessionAuthorization()))
        await _wait_for_file(leader_file)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task  # raised only after the worker has stopped and reaped git
        # Checked here, inside the loop, before asyncio.run's shutdown can wait for the worker thread
        leader, child = int(leader_file.read_text()), int(child_file.read_text())
        with pytest.raises(ProcessLookupError):  # reaped by us: its pid no longer exists at all
            os.kill(leader, 0)
        _assert_gone(child)  # the background child shared the group; it was signalled too
        assert list(scratch_root.glob("agent-loop-git-*")) == [], "temporary index was not removed"

    started = time.monotonic()
    asyncio.run(cancel_once_git_is_running())
    assert time.monotonic() - started < 10, "cancellation waited for the git time limit"


@needs_git
def test_cancellation_does_not_touch_unrelated_processes(repo, fake_git_with_child, monkeypatch):
    leader_file, _ = fake_git_with_child
    monkeypatch.setattr(git_tools, "GIT_TIMEOUT_SECONDS", 60)
    bystander = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        async def cancel_once_git_is_running():
            task = asyncio.create_task(dispatch.dispatch_tool_async("git_status", {},
                                                                    authorization=SessionAuthorization()))
            await _wait_for_file(leader_file)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert bystander.poll() is None and _alive(bystander.pid), "an unrelated process was stopped"
            assert os.getpgid(bystander.pid) != int(leader_file.read_text())

        asyncio.run(cancel_once_git_is_running())
    finally:
        bystander.kill()
        bystander.wait()


@needs_git
def test_run_command_still_asks_for_git_and_git_tools_do_not_need_a_grant(repo):
    asked = asyncio.run(dispatch.dispatch_tool_async(
        "run_command", {"command": "git status"}, authorization=SessionAuthorization()))
    assert asked.get("requiresPermission") is True
    assert asked["kind"] == "command" and asked["identifier"] == "git"

    direct = asyncio.run(dispatch.dispatch_tool_async("git_status", {}, authorization=SessionAuthorization()))
    assert direct["isError"] is False and "requiresPermission" not in direct


@needs_git
def test_skill_allowlist_stops_git_before_it_runs(repo, tmp_path, monkeypatch):
    marker = tmp_path / "git-ran"
    recorder = tmp_path / "recording-git"
    recorder.write_text(f"#!/bin/sh\necho ran >> '{marker}'\nexec '{REAL_GIT}' \"$@\"\n", encoding="utf-8")
    recorder.chmod(0o755)
    monkeypatch.setattr(git_tools, "_git_executable", lambda: str(recorder))
    for name in ("git_status", "git_diff"):
        result = asyncio.run(dispatch.dispatch_tool_async(
            name, {}, allowed_tools=["read_file"], authorization=SessionAuthorization(allow_all=True)))
        assert result["isError"] is True and "not permitted" in result["content"], name
    assert not marker.exists(), "git ran although the skill allowlist excluded it"


def test_git_tools_are_registered_as_git_kind_with_closed_schemas():
    registry = local_registry()
    assert registry.kind_of("git_status") == GIT and registry.kind_of("git_diff") == GIT
    status = registry.get("git_status").schema["function"]["parameters"]
    diff = registry.get("git_diff").schema["function"]["parameters"]
    assert status["properties"] == {} and status["required"] == []
    assert set(diff["properties"]) == {"path"} and diff["required"] == []


def test_the_git_module_reads_no_user_configuration():
    assert git_tools._git_env()["GIT_CONFIG_GLOBAL"] == os.devnull
    assert git_tools._git_env()["GIT_CONFIG_NOSYSTEM"] == "1"
    assert "core.fsmonitor=false" in git_tools._GLOBAL_OPTIONS
    assert "--no-optional-locks" in git_tools._GLOBAL_OPTIONS
