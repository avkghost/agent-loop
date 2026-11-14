"""Scripted-provider integration: read a file, make a targeted edit, inspect the diff, run an authorized test command,
and report. Only the model is scripted. The loop, dispatcher, file tools, Git inspection and shell run for real, in a
temporary repository. No network, no real model."""

import asyncio
import json
import os
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from src import loop
from tests.conftest import protocol_violations

REAL_GIT = shutil.which("git")
needs_git = pytest.mark.skipif(REAL_GIT is None, reason="git is not installed in this environment")
NO_OUTPUT = "[Command completed with no visual output]"


def _init_repo(workdir):
    env = dict(os.environ, GIT_AUTHOR_NAME="Test", GIT_AUTHOR_EMAIL="test@example.invalid",
               GIT_COMMITTER_NAME="Test", GIT_COMMITTER_EMAIL="test@example.invalid",
               GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    (workdir / "app.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    (workdir / "check_app.py").write_text(
        "import sys\nimport app\nsys.exit(0 if app.value() == 2 else 1)\n", encoding="utf-8")
    for args in (["init", "-q", "-b", "main"], ["add", "app.py", "check_app.py"], ["commit", "-q", "-m", "initial"]):
        subprocess.run([REAL_GIT, *args], cwd=workdir, env=env, check=True, capture_output=True)


def _call(call_id, name, arguments):
    return {"id": call_id, "name": name, "arguments": json.dumps(arguments)}


def _run_goal(goal):
    event_loop = asyncio.new_event_loop()
    try:
        return event_loop.run_until_complete(
            loop.agent_loop_async(goal, SimpleNamespace(get_openai_tools=lambda: []), event_loop))
    finally:
        event_loop.close()


@pytest.fixture
def project(tmp_path, monkeypatch, tmp_sessions):
    workdir = tmp_path / "project"
    workdir.mkdir()
    _init_repo(workdir)
    monkeypatch.chdir(workdir)
    monkeypatch.setattr(loop, "load_project_guidelines", lambda: None)
    return workdir


@needs_git
@pytest.mark.real_process
def test_read_edit_diff_test_and_report(project, fake_post, streamed_response):
    script = [
        streamed_response(finish_reason="tool_calls",
                          tool_calls=[_call("c1", "read_file", {"path": "app.py"})]),
        streamed_response(finish_reason="tool_calls", tool_calls=[_call("c2", "edit_file", {
            "path": "app.py", "old_string": "    return 1\n", "new_string": "    return 2\n"})]),
        streamed_response(finish_reason="tool_calls", tool_calls=[_call("c3", "git_diff", {})]),
        streamed_response(finish_reason="tool_calls",
                          tool_calls=[_call("c4", "run_command", {"command": "python3 check_app.py"})]),
        streamed_response(content="Changed value() to return 2. The diff shows only that line, and "
                                  "check_app.py exits 0."),
    ]
    post = fake_post(script)

    _run_goal("Make app.value() return 2 and prove it with check_app.py.")

    assert len(post.payloads) == 5
    messages = post.payloads[-1]["messages"]
    assert protocol_violations(messages) == []
    results = {m["tool_call_id"]: m["content"] for m in messages if m["role"] == "tool"}

    assert "return 1" in results["c1"]
    assert "Edited app.py" in results["c2"] and "at line 2" in results["c2"]
    assert "Comparing the working tree with HEAD" in results["c3"]
    assert "-    return 1" in results["c3"] and "+    return 2" in results["c3"]
    assert "requiresPermission" not in results["c4"]
    assert NO_OUTPUT in results["c4"], "check_app.py did not run to a clean exit"
    assert (project / "app.py").read_text(encoding="utf-8") == "def value():\n    return 2\n"


@pytest.mark.real_process
def test_a_refused_ambiguous_edit_is_reported_and_the_model_recovers(project, fake_post, streamed_response):
    (project / "app.py").write_text("def value():\n    return 1\n\ndef other():\n    return 1\n", encoding="utf-8")
    script = [
        streamed_response(finish_reason="tool_calls", tool_calls=[_call("c1", "edit_file", {
            "path": "app.py", "old_string": "    return 1\n", "new_string": "    return 2\n"})]),
        streamed_response(finish_reason="tool_calls", tool_calls=[_call("c2", "edit_file", {
            "path": "app.py", "old_string": "def value():\n    return 1\n",
            "new_string": "def value():\n    return 2\n"})]),
        streamed_response(content="The first edit matched twice and changed nothing; the second, with "
                                  "more context, changed value()."),
    ]
    post = fake_post(script)

    _run_goal("Make value() return 2 without touching other().")

    messages = post.payloads[-1]["messages"]
    assert protocol_violations(messages) == []
    results = {m["tool_call_id"]: m["content"] for m in messages if m["role"] == "tool"}
    assert "more than one location" in results["c1"]
    assert "Edited app.py" in results["c2"]
    assert (project / "app.py").read_text(encoding="utf-8") == (
        "def value():\n    return 2\n\ndef other():\n    return 1\n")
