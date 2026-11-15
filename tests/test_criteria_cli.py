"""Phase C2: trusted acceptance criteria through the CLI.

Criteria come only from a validated document given on the command line. These tests drive the real CLI, the real
loop, the real verification engine and the real dispatcher with a scripted model. The model's replies are the only
simulated part. Nothing reaches a real endpoint.
"""

import json
import os
import shutil
import subprocess

import pytest

from src import acceptance, cli, config, envelope, loop
from tests import scripted_provider
from tests.conftest import make_streamed_response

REAL_GIT = shutil.which("git")
needs_git = pytest.mark.skipif(REAL_GIT is None, reason="git is not installed in this environment")


@pytest.fixture
def project(tmp_path, monkeypatch, tmp_sessions):
    workdir = tmp_path / "project"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    monkeypatch.setattr(loop, "load_project_guidelines", lambda: None)
    (workdir / "app.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    return workdir


@pytest.fixture
def model(monkeypatch):
    """A scripted model. Every request is recorded. The provider turns an exception into a provider failure, so a
    request beyond the script is also recorded here and fails the test on teardown."""
    class Model:
        def __init__(self):
            self.replies = []
            self.payloads = []
            self.unexpected = 0

        def script(self, *replies):
            self.replies = list(replies)

        def __call__(self, *args, **kwargs):
            self.payloads.append(json.loads(json.dumps(kwargs.get("json"))))
            if not self.replies:
                self.unexpected += 1
                raise AssertionError("the model was asked for a reply the script does not provide")
            return self.replies.pop(0)

    scripted = Model()
    monkeypatch.setattr(scripted_provider, "POST", scripted)
    yield scripted
    assert scripted.unexpected == 0, "the run made a model request the test did not script"


def _checker(project):
    """A verification command that needs no shell operators: it exits 0 only when app.value() is 2."""
    (project / "check_value.py").write_text("import sys\nimport app\nsys.exit(0 if app.value() == 2 else 1)\n",
                                            encoding="utf-8")


def _stop(text):
    return make_streamed_response(finish_reason="stop", content=text)


def _call(call_id, name, arguments):
    return make_streamed_response(finish_reason="tool_calls", tool_calls=[
        {"id": call_id, "name": name, "arguments": json.dumps(arguments)}])


def _doc(*criteria):
    return json.dumps({"criteria": list(criteria)})


FILE_HAS_TWO = {"criterion_id": "app_returns_two", "criterion_type": "file_content", "expected": True,
                "parameters": {"path": "app.py", "contains": "return 2"}}
TESTS_PASS = {"criterion_id": "check_passes", "criterion_type": "command_exit", "expected": 0,
              "parameters": {"command": "python3 check_value.py"}}


def _write(directory, name, text):
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def _session_file(tmp_sessions, session_id):
    return json.loads((tmp_sessions / f"{session_id}.json").read_text(encoding="utf-8"))


def _only_session_id(tmp_sessions):
    files = sorted(tmp_sessions.glob("*.json"))
    assert len(files) == 1, files
    return files[0].stem


# --- 1. valid input -----------------------------------------------------------------------------------------------

def test_a_valid_document_parses_into_user_origin_criteria_without_a_run_id():
    parsed = acceptance.parse_criteria_document(_doc(FILE_HAS_TWO, TESTS_PASS), max_criteria=20)
    assert [c.criterion_id for c in parsed] == ["app_returns_two", "check_passes"]
    assert all(c.origin == "user" and c.run_id == "" and c.trusted for c in parsed)


# --- 2-4. invalid input, refused before any model request --------------------------------------------------------

@pytest.mark.parametrize("text, fragment", [
    ("not json at all", "not valid JSON"),
    ('{"criteria": [1, 2]', "not valid JSON"),
    ('{"criteria": []}', "non-empty list"),
    ('{"items": []}', 'only key is "criteria"'),
    ('{"criteria": [{"criterion_id": "a", "criterion_type": "file_exists", "expected": true}]}', "missing fields"),
    ('{"criteria": [{"criterion_id": "a", "criterion_type": "file_exists", "expected": true, '
     '"parameters": {"path": "x"}, "origin": "model"}]}', "set by the runtime"),
    ('{"criteria": [{"criterion_id": "a", "criterion_type": "file_exists", "expected": "yes", '
     '"parameters": {"path": "x"}}]}', "must be true (present) or false"),
    ('{"criteria": [{"criterion_id": "a", "criterion_type": "file_content", "expected": true, '
     '"parameters": {"path": "x", "contains": "y", "pattern": "z"}}]}', "unknown fields"),
    ('{"criteria": [{"criterion_id": "a", "criterion_type": "command_exit", "expected": 0, '
     '"parameters": {"command": "python3 x.py", "timeout_seconds": 0}}]}', "timeout_seconds"),
    ('{"criteria": [{"criterion_id": "a", "criterion_type": "file_exists", "expected": true, '
     '"parameters": {"path": "x"}}, {"criterion_id": "a", "criterion_type": "file_exists", "expected": true, '
     '"parameters": {"path": "y"}}]}', "used more than once"),
    ('{"criteria": [{"criterion_id": "a", "criterion_type": "file_exists", "expected": true, '
     '"parameters": {"path": "x"}}], "criteria": []}', "duplicate key"),
])
def test_malformed_documents_are_refused_with_a_located_reason(text, fragment):
    with pytest.raises(acceptance.CriterionDocumentError) as info:
        acceptance.parse_criteria_document(text, max_criteria=20)
    assert any(fragment in problem for problem in info.value.problems), info.value.problems


def test_more_criteria_than_the_engine_can_check_are_refused_not_truncated():
    many = [{"criterion_id": f"c{i}", "criterion_type": "file_exists", "expected": True,
             "parameters": {"path": f"f{i}.txt"}} for i in range(21)]
    with pytest.raises(acceptance.CriterionDocumentError, match="at most 20"):
        acceptance.parse_criteria_document(json.dumps({"criteria": many}), max_criteria=20)


# --- 3. unsupported criteria --------------------------------------------------------------------------------------

@pytest.mark.parametrize("criterion_type", ["job_state", "llm_judge", "shell_exec"])
def test_unsupported_or_undeclarable_types_are_refused(criterion_type):
    entry = {"criterion_id": "a", "criterion_type": criterion_type, "expected": "running",
             "parameters": {"job_id": "x"}}
    with pytest.raises(acceptance.CriterionDocumentError) as info:
        acceptance.parse_criteria_document(_doc(entry), max_criteria=20)
    assert any("criterion_type" in p or "job_state" in p for p in info.value.problems)


# --- 4-5. validation before execution: unsafe paths and commands -------------------------------------------------

def test_paths_outside_the_workspace_and_unauthorized_commands_are_refused_before_any_run(project, model, capsys,
                                                                                         tmp_sessions):
    escape = {"criterion_id": "escape", "criterion_type": "file_exists", "expected": True,
              "parameters": {"path": "../secret.txt"}}
    piped = {"criterion_id": "piped", "criterion_type": "command_exit", "expected": 0,
             "parameters": {"command": "python3 app.py | head"}}
    needs_approval = {"criterion_id": "approval", "criterion_type": "command_exit", "expected": 0,
                      "parameters": {"command": "uname -s"}}
    path = _write(project, "criteria.json", _doc(escape, piped, needs_approval))
    cli._main(["run", "goal", "--criteria", path])
    out = capsys.readouterr().out
    assert "could never run" in out and "nothing was run" in out
    assert "path is outside the working directory" in out
    assert out.count("it needs approval, and verification never prompts") == 2
    assert model.payloads == [], "a refused criteria document must not reach the model"
    assert list(tmp_sessions.glob("*.json")) == [], "no session may be created for a refused document"


def test_a_missing_or_undecodable_criteria_file_runs_nothing(project, model, capsys, tmp_sessions):
    cli._main(["run", "goal", "--criteria", str(project / "absent.json")])
    assert "Cannot read criteria file" in capsys.readouterr().out
    bad = project / "binary.json"
    bad.write_bytes(b"\xff\xfe\x00")
    cli._main(["run", "goal", "--criteria", str(bad)])
    assert "not UTF-8" in capsys.readouterr().out
    assert model.payloads == [] and list(tmp_sessions.glob("*.json")) == []


@pytest.mark.parametrize("cycles", ["0", "11", "two"])
def test_invalid_verification_cycles_are_refused(project, model, capsys, cycles):
    if not cycles.isdigit():
        with pytest.raises(SystemExit):
            cli.parse_args(["run", "goal", "--verification-cycles", cycles])
        return
    cli._main(["run", "goal", "--verification-cycles", cycles])
    assert "--verification-cycles must be an integer from 1 to 10" in capsys.readouterr().out
    assert model.payloads == []


# --- 6-9. verification outcomes, repair and exhaustion (through the CLI) --------------------------------------------

def test_a_passing_criteria_run_is_verified_and_reports_each_criterion(project, model, capsys, tmp_sessions):
    (project / "app.py").write_text("def value():\n    return 2\n", encoding="utf-8")
    _checker(project)
    path = _write(project, "criteria.json", _doc(FILE_HAS_TWO, TESTS_PASS))
    model.script(_stop("value() now returns 2."))
    cli._main(["run", "make value two", "--criteria", path])
    out = capsys.readouterr().out
    assert "VERIFIED" in out and "all 2 trusted criteria passed" in out
    assert "app_returns_two [file_content]: passed" in out
    assert "check_passes [command_exit]: passed" in out


def test_a_failing_run_with_an_exhausted_budget_is_reported_as_failed_verification(project, model, capsys, tmp_sessions):
    path = _write(project, "criteria.json", _doc(FILE_HAS_TWO))
    model.script(_stop("Done."))
    cli._main(["run", "make value two", "--criteria", path, "--verification-cycles", "1"])
    out = capsys.readouterr().out
    assert "FAILED VERIFICATION" in out and "VERIFIED" not in out
    assert "Task completed successfully" not in out, "a failed verification must not be announced as success"
    assert "app_returns_two [file_content]: failed (attempt 1 of 1)" in out
    assert "expected content was not observed" in out


def test_a_failed_check_is_returned_to_the_model_and_a_bounded_repair_verifies(project, model, capsys, tmp_sessions):
    path = _write(project, "criteria.json", _doc(FILE_HAS_TWO))
    model.script(
        _stop("Done."),                                                                    # attempt 1 fails
        _call("r1", "edit_file", {"path": "app.py", "old_string": "return 1", "new_string": "return 2"}),
        _stop("Repaired."),                                                                # attempt 2 passes
    )
    cli._main(["run", "make value two", "--criteria", path])
    out = capsys.readouterr().out
    assert "VERIFIED" in out and "app_returns_two [file_content]: passed (attempt 2 of 2)" in out
    feedback = [m["content"] for m in model.payloads[1]["messages"] if m["role"] == "user"]
    assert any("Verification failed" in text and "app_returns_two" in text for text in feedback)
    saved = _session_file(tmp_sessions, _only_session_id(tmp_sessions))
    attempts = sorted(r["attempt"] for r in saved["verification_results"])
    assert attempts == [1, 2], "the failed attempt stays in the history"


# --- 10-11. persistence and resume ----------------------------------------------------------------------------------

def test_the_trusted_contract_is_persisted_with_the_session(project, model, capsys, tmp_sessions):
    path = _write(project, "criteria.json", _doc(FILE_HAS_TWO))
    model.script(_stop("Done."))
    cli._main(["run", "goal", "--criteria", path, "--verification-cycles", "1"])
    saved = _session_file(tmp_sessions, _only_session_id(tmp_sessions))
    assert [c["criterion_id"] for c in saved["criteria"]] == ["app_returns_two"]
    assert saved["criteria"][0]["origin"] == "user" and saved["criteria"][0]["parameters"]["contains"] == "return 2"


def _finished_verified_session(project, model, capsys, tmp_sessions, path):
    (project / "app.py").write_text("def value():\n    return 2\n", encoding="utf-8")
    model.script(_stop("Done."))
    cli._main(["run", "first goal", "--criteria", path])
    capsys.readouterr()
    return _only_session_id(tmp_sessions)


def test_resume_without_criteria_carries_the_contract_over_and_verifies_the_new_run(project, model, capsys, tmp_sessions):
    path = _write(project, "criteria.json", _doc(FILE_HAS_TWO))
    sid = _finished_verified_session(project, model, capsys, tmp_sessions, path)
    model.script(_stop("Still two."))
    cli._main(["resume", sid])
    out = capsys.readouterr().out
    assert "Trusted criteria carried over from the session: 1 check(s)" in out
    assert "VERIFIED" in out and "app_returns_two:r1 [file_content]: passed" in out
    saved = _session_file(tmp_sessions, sid)
    assert len(saved["criteria"]) == 2
    assert {c["criterion_id"] for c in saved["criteria"]} == {"app_returns_two", "app_returns_two:r1"}


def test_resume_with_the_same_criteria_is_accepted(project, model, capsys, tmp_sessions):
    path = _write(project, "criteria.json", _doc(FILE_HAS_TWO))
    sid = _finished_verified_session(project, model, capsys, tmp_sessions, path)
    model.script(_stop("Still two."))
    cli._main(["resume", sid, "--criteria", path])
    assert "VERIFIED" in capsys.readouterr().out


def test_resume_with_modified_criteria_is_refused_and_nothing_runs(project, model, capsys, tmp_sessions):
    path = _write(project, "criteria.json", _doc(FILE_HAS_TWO))
    sid = _finished_verified_session(project, model, capsys, tmp_sessions, path)
    weaker = {"criterion_id": "app_returns_two", "criterion_type": "file_exists", "expected": True,
              "parameters": {"path": "app.py"}}
    changed = _write(project, "weaker.json", _doc(weaker))
    before = _session_file(tmp_sessions, sid)
    requests_before = len(model.payloads)
    cli._main(["resume", sid, "--criteria", changed])
    out = capsys.readouterr().out
    assert "differ from the session's trusted criteria" in out and "nothing was run" in out
    assert len(model.payloads) == requests_before, "a refused resume must not reach the model"
    assert _session_file(tmp_sessions, sid) == before


def test_criteria_cannot_be_added_when_the_session_never_had_them(project, model, capsys, tmp_sessions):
    model.script(_stop("Done."))
    cli._main(["run", "plain goal"])
    capsys.readouterr()
    sid = _only_session_id(tmp_sessions)
    path = _write(project, "criteria.json", _doc(FILE_HAS_TWO))
    calls = len(model.payloads)
    cli._main(["resume", sid, "--criteria", path])
    assert "cannot be added on resume" in capsys.readouterr().out
    assert len(model.payloads) == calls


def test_a_renamed_identifier_with_identical_checks_is_the_same_contract(project, model, capsys, tmp_sessions):
    path = _write(project, "criteria.json", _doc(FILE_HAS_TWO))
    sid = _finished_verified_session(project, model, capsys, tmp_sessions, path)
    renamed = dict(FILE_HAS_TWO, criterion_id="renamed_label")
    model.script(_stop("Still two."))
    cli._main(["resume", sid, "--criteria", _write(project, "renamed.json", _doc(renamed))])
    assert "VERIFIED" in capsys.readouterr().out


def test_resumed_identifiers_that_would_exceed_the_limit_are_refused(project, model, capsys, tmp_sessions):
    long_id = dict(FILE_HAS_TWO, criterion_id="x" * 62)  # + ":r1" exceeds the 64-character limit
    path = _write(project, "criteria.json", _doc(long_id))
    sid = _finished_verified_session(project, model, capsys, tmp_sessions, path)
    calls = len(model.payloads)
    cli._main(["resume", sid])
    assert "cannot be carried into a resumed run" in capsys.readouterr().out
    assert len(model.payloads) == calls


# --- 12. UNKNOWN recovery: an unresolved outcome withholds verified completion; resume never redispatches ---------

def test_an_unknown_tool_outcome_withholds_verified_completion_and_resume_does_not_redispatch(project, model, capsys,
                                                                                            tmp_sessions, monkeypatch):
    calls = []

    async def unknown_dispatch(tool_name, tool_input, *args, **kwargs):
        # A tool that lost its outcome reports unknown_outcome; the loop records it and does not retry the call
        calls.append(tool_name)
        return envelope.error_unknown_outcome("transport lost after the request was sent; not retried")

    monkeypatch.setattr(loop.dispatch, "dispatch_tool_async", unknown_dispatch)
    path = _write(project, "criteria.json", _doc(FILE_HAS_TWO))
    model.script(_call("u1", "write_file", {"path": "app.py", "content": "def value():\n    return 2\n"}), _stop("Done."))
    cli._main(["run", "goal", "--criteria", path])
    out = capsys.readouterr().out
    assert "VERIFIED" not in out and "NEEDS ATTENTION" in out
    assert "unknown, so verified completion is withheld" in out
    assert calls == ["write_file"]
    sid = _only_session_id(tmp_sessions)
    model.script(_stop("Carry on."))
    cli._main(["resume", sid, "--verification-cycles", "1"])
    capsys.readouterr()
    assert calls == ["write_file"], "an ambiguous side effect must not be redispatched on resume"


# --- 13. unauthorized verification operations --------------------------------------------------------------------------

def test_a_verification_command_that_would_prompt_is_never_run_and_cannot_pass(project, model, capsys, tmp_sessions):
    """A declared command is validated by the same rules at declaration time: a command needing approval is refused
    before the run, so it can never be run under the runtime's prompt path."""
    needs_prompt = {"criterion_id": "p", "criterion_type": "command_exit", "expected": 0,
                    "parameters": {"command": "python3 app.py > out.txt"}}
    path = _write(project, "criteria.json", _doc(needs_prompt))
    cli._main(["run", "goal", "--criteria", path])
    assert "could never run" in capsys.readouterr().out
    assert model.payloads == [] and not (project / "out.txt").exists()


# --- 14. model influence on trusted criteria ----------------------------------------------------------------------------

def test_a_model_cannot_add_or_alter_trusted_criteria_through_tool_calls_or_files(project, model, capsys, tmp_sessions):
    path = _write(project, "criteria.json", _doc(FILE_HAS_TWO))
    model.script(
        _call("m1", "add_criterion", {"criterion_id": "trivial", "criterion_type": "file_exists", "expected": True,
                                       "parameters": {"path": "app.py"}}),
        _call("m2", "write_file", {"path": "criteria.json", "content": _doc(
            {"criterion_id": "app_returns_two", "criterion_type": "file_exists", "expected": True,
             "parameters": {"path": "app.py"}})}),
        _stop("All criteria pass."),
        _stop("Still nothing changed."),
    )
    cli._main(["run", "goal", "--criteria", path, "--verification-cycles", "2"])
    out = capsys.readouterr().out
    assert "FAILED VERIFICATION" in out, out
    saved = _session_file(tmp_sessions, _only_session_id(tmp_sessions))
    assert [c["criterion_id"] for c in saved["criteria"]] == ["app_returns_two"]
    assert saved["criteria"][0]["parameters"] == {"path": "app.py", "contains": "return 2"}
    assert {r["criterion_id"] for r in saved["verification_results"]} == {"app_returns_two"}


# --- reporting without criteria ----------------------------------------------------------------------------------------

def test_a_run_without_criteria_reports_that_nothing_was_verified(project, model, capsys, tmp_sessions):
    model.script(_stop("Finished my work."))
    cli._main(["run", "plain goal"])
    out = capsys.readouterr().out
    assert "COMPLETED WITHOUT TRUSTED CRITERIA" in out and "nothing was verified" in out
    assert "VERIFIED" not in out
    assert "Task completed successfully! (not verified)" in out


# --- scripted-provider integration: read, edit, git_diff, authorized test command, trusted verification -------------

@needs_git
@pytest.mark.real_process
def test_read_edit_diff_test_then_trusted_verification_through_the_cli(project, model, capsys, tmp_sessions):
    env = dict(os.environ, GIT_AUTHOR_NAME="Test", GIT_AUTHOR_EMAIL="test@example.invalid",
               GIT_COMMITTER_NAME="Test", GIT_COMMITTER_EMAIL="test@example.invalid",
               GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    (project / "check_app.py").write_text("import sys\nimport app\nsys.exit(0 if app.value() == 2 else 1)\n",
                                          encoding="utf-8")
    for args in (["init", "-q", "-b", "main"], ["add", "app.py", "check_app.py"], ["commit", "-q", "-m", "initial"]):
        subprocess.run([REAL_GIT, *args], cwd=project, env=env, check=True, capture_output=True)
    criteria = _write(project, "criteria.json", _doc(FILE_HAS_TWO, {
        "criterion_id": "check_passes", "criterion_type": "command_exit", "expected": 0,
        "parameters": {"command": "python3 check_app.py"}}))
    model.script(
        _call("c1", "read_file", {"path": "app.py"}),
        _call("c2", "edit_file", {"path": "app.py", "old_string": "    return 1\n", "new_string": "    return 2\n"}),
        _call("c3", "git_diff", {}),
        _call("c4", "run_command", {"command": "python3 check_app.py"}),
        _stop("Changed value() to return 2; the diff shows only that line and check_app.py exits 0."),
    )
    cli._main(["run", "make value() return 2", "--criteria", criteria])
    out = capsys.readouterr().out
    assert "VERIFIED" in out and "all 2 trusted criteria passed" in out
    messages = model.payloads[-1]["messages"]
    results = {m["tool_call_id"]: m["content"] for m in messages if m["role"] == "tool"}
    assert "return 1" in results["c1"] and "Edited app.py" in results["c2"]
    assert "+    return 2" in results["c3"]
    assert "[Command completed with no visual output]" in results["c4"]
    assert (project / "app.py").read_text(encoding="utf-8") == "def value():\n    return 2\n"


# --- review: restored criteria are judged by the current configuration; unknown outcomes stay visible ------------

def test_resume_refuses_restored_checks_that_the_current_configuration_no_longer_allows(project, model, capsys,
                                                                                       tmp_sessions, monkeypatch):
    check = {"criterion_id": "tests_pass", "criterion_type": "command_exit", "expected": 0,
             "parameters": {"command": "python3 -c \"print(1)\""}}
    path = _write(project, "criteria.json", _doc(check))
    model.script(_stop("Done."))
    cli._main(["run", "first goal", "--criteria", path, "--verification-cycles", "1"])
    capsys.readouterr()
    sid = _only_session_id(tmp_sessions)
    before = _session_file(tmp_sessions, sid)
    monkeypatch.setattr(config, "ALLOWED_TOOLS", set())  # python3 is no longer allowed
    requests_before = len(model.payloads)
    cli._main(["resume", sid, "--verification-cycles", "1"])
    out = capsys.readouterr().out
    assert "not valid under the current configuration; nothing was run" in out
    assert "tests_pass: verification command is not allowed" in out
    assert len(model.payloads) == requests_before, "a refused resume must not reach the model"
    assert _session_file(tmp_sessions, sid) == before


def test_an_earlier_unknown_outcome_is_still_reported_when_the_resumed_run_verifies(project, model, capsys,
                                                                                    tmp_sessions, monkeypatch):
    from src import envelope, loop
    original = loop.dispatch.dispatch_tool_async

    async def unknown_dispatch(tool_name, tool_input, *args, **kwargs):
        return envelope.error_unknown_outcome("transport lost; not retried")

    monkeypatch.setattr(loop.dispatch, "dispatch_tool_async", unknown_dispatch)
    path = _write(project, "criteria.json", _doc(FILE_HAS_TWO))
    model.script(_call("u1", "write_file", {"path": "app.py", "content": "def value():\n    return 2\n"}), _stop("Done."))
    cli._main(["run", "first", "--criteria", path, "--verification-cycles", "1"])
    capsys.readouterr()
    sid = _only_session_id(tmp_sessions)
    monkeypatch.setattr(loop.dispatch, "dispatch_tool_async", original)
    (project / "app.py").write_text("def value():\n    return 2\n", encoding="utf-8")
    model.script(_stop("Carry on."))
    cli._main(["resume", sid, "--verification-cycles", "1"])
    out = capsys.readouterr().out
    assert "VERIFIED — all 1 trusted criteria passed" in out, "the verdict itself is not redefined"
    assert "1 tool outcome(s) in this session are recorded as unknown and were not retried" in out
    assert "This verdict does not resolve them." in out


def test_a_verification_command_runs_as_allowed_and_may_change_the_workspace(project, model, capsys, tmp_sessions):
    """Documented trust boundary: verification uses the run_command allowlist, so an allowed command can change files.
    The criteria file cannot be changed by the model; what the command does is the operator's responsibility."""
    doomed = {"criterion_id": "removes_file", "criterion_type": "command_exit", "expected": 0,
              "parameters": {"command": "rm app.py"}}
    path = _write(project, "criteria.json", _doc(doomed))
    model.script(_stop("Done."))
    cli._main(["run", "goal", "--criteria", path, "--verification-cycles", "1"])
    assert "VERIFIED" in capsys.readouterr().out
    assert not (project / "app.py").exists()


def test_a_check_script_written_by_the_model_is_run_as_it_is(project, model, capsys, tmp_sessions):
    """Documented limit: a command_exit check is only independent of the model if the code it runs is not the model's.
    Here the model writes the checker, and the criterion passes."""
    check = {"criterion_id": "impl_passes", "criterion_type": "command_exit", "expected": 0,
             "parameters": {"command": "python3 check.py"}}
    path = _write(project, "criteria.json", _doc(check))
    model.script(_call("w1", "write_file", {"path": "check.py", "content": "import sys\nsys.exit(0)\n"}), _stop("Done."))
    cli._main(["run", "goal", "--criteria", path, "--verification-cycles", "1"])
    assert "VERIFIED" in capsys.readouterr().out
