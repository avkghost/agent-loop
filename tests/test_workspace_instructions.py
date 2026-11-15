"""Workspace instruction discovery and loading: AGENTS.md entrypoints, explicit local links, bounds, and boundaries.

Offline only. Projects and the home directory are temporary; the scripted provider answers any model request.
"""

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import instructions, loop, runs
from src.tools import get_local_tools
from src.tools.files import read_file_safe
from tests import scripted_provider
from tests.conftest import make_streamed_response

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def home(tmp_path, monkeypatch):
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    return fake_home


def project(tmp_path, name="project"):
    root = tmp_path / name
    (root / ".git").mkdir(parents=True)
    return root


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def discover(start: Path, home_dir: Path):
    return instructions.discover(start.resolve(), home_dir)


# --- discovery ----------------------------------------------------------------------------------------------------

def test_a_root_workspace_loads_its_entrypoint(tmp_path, home):
    root = project(tmp_path)
    write(root / "AGENTS.md", "Root rules.")
    found = discover(root, home)
    assert found.files == ("AGENTS.md",)
    assert "Root rules." in found.text


def test_a_nested_workspace_inherits_the_parent_entrypoint_and_keeps_its_directory(tmp_path, home, monkeypatch):
    root = project(tmp_path)
    write(root / "AGENTS.md", "Parent rules.")
    nested = root / "tmp" / "test"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    before = sorted(p.name for p in nested.iterdir())
    found = discover(Path.cwd(), home)
    assert found.root == root.resolve() and "Parent rules." in found.text
    assert Path.cwd() == nested.resolve()
    assert sorted(p.name for p in nested.iterdir()) == before, "discovery must not write into the workspace"


def test_the_nearest_entrypoint_takes_precedence_and_parents_are_not_merged(tmp_path, home):
    root = project(tmp_path)
    write(root / "AGENTS.md", "Parent rules.")
    nested = write(root / "tmp" / "test" / "AGENTS.md", "Nested rules.").parent
    found = discover(nested, home)
    assert found.files == ("AGENTS.md",)
    assert "Nested rules." in found.text and "Parent rules." not in found.text


def test_discovery_stops_at_the_project_root_and_never_loads_home_instructions(tmp_path, home):
    write(home / "AGENTS.md", "Home rules.")
    root = project(tmp_path)
    nested = (root / "work")
    nested.mkdir()
    assert discover(nested, home) is None, "a parent above the project root is not searched"
    assert discover(home, home) is None, "the home directory itself is never searched"
    workspace_in_home = home / "scratch"
    workspace_in_home.mkdir()
    assert discover(workspace_in_home, home) is None, "home instructions are not inherited by a home workspace"


# --- explicit links ------------------------------------------------------------------------------------------------

def test_the_entrypoint_links_to_wo_and_context_mode(tmp_path, home):
    root = project(tmp_path)
    write(root / "AGENTS.md", "- [Workflow](WO.md)\n- [Routing](instructions/context-mode.md)\n")
    write(root / "WO.md", "Work orchestration rules.")
    write(root / "instructions" / "context-mode.md", "Routing rules.")
    found = discover(root, home)
    assert found.files == ("AGENTS.md", "WO.md", "instructions/context-mode.md")
    assert found.text.index("Work orchestration rules.") < found.text.index("Routing rules.")


def test_the_real_repository_entrypoint_resolves_to_its_linked_documents(home):
    found = instructions.discover(REPO, home)
    assert found.files == ("AGENTS.md", "WO.md", "instructions/context-mode.md")


def test_nested_references_are_followed_relative_to_the_referring_file(tmp_path, home):
    root = project(tmp_path)
    write(root / "AGENTS.md", "[A](docs/a.md)")
    write(root / "docs" / "a.md", "A text. [B](b.md)")
    write(root / "docs" / "b.md", "B text. [C](../notes/c.md)")
    write(root / "notes" / "c.md", "C text.")
    found = discover(root, home)
    assert found.files == ("AGENTS.md", "docs/a.md", "docs/b.md", "notes/c.md")


def test_duplicate_references_load_once_and_cycles_are_detected_without_reloading(tmp_path, home):
    root = project(tmp_path)
    write(root / "AGENTS.md", "[A](a.md) [B](b.md) [A again](a.md)")
    write(root / "a.md", "A. [B](b.md)")
    write(root / "b.md", "B. [A](a.md)")
    found = discover(root, home)
    assert found.files == ("AGENTS.md", "a.md", "b.md")
    assert found.text.count("A. ") == 1 and found.text.count("B. ") == 1
    assert found.cycles == ("AGENTS.md -> a.md -> b.md -> a.md",)


def test_a_missing_required_document_is_reported_by_path(tmp_path, home):
    root = project(tmp_path)
    write(root / "AGENTS.md", "[Workflow](WO.md)")
    with pytest.raises(instructions.InstructionError, match="missing: WO.md"):
        discover(root, home)


def test_path_traversal_out_of_the_instruction_root_is_rejected(tmp_path, home):
    root = project(tmp_path)
    write(tmp_path / "outside.md", "secret")
    write(root / "AGENTS.md", "[Outside](../outside.md)")
    with pytest.raises(instructions.InstructionError, match="outside the instruction root"):
        discover(root, home)


def test_absolute_and_remote_instruction_sources_are_rejected(tmp_path, home):
    root = project(tmp_path)
    write(root / "AGENTS.md", "[Abs](/etc/rules.md)")
    with pytest.raises(instructions.InstructionError, match="absolute paths"):
        discover(root, home)
    write(root / "AGENTS.md", "[Remote](https://example.com/rules.md)")
    with pytest.raises(instructions.InstructionError, match="remote instruction sources"):
        discover(root, home)


def test_symbolic_link_escapes_are_rejected(tmp_path, home):
    root = project(tmp_path)
    outside = write(tmp_path / "outside-rules.md", "secret rules")
    write(root / "AGENTS.md", "[Leak](instructions/leak.md)")
    (root / "instructions").mkdir()
    os.symlink(outside, root / "instructions" / "leak.md")
    with pytest.raises(instructions.InstructionError, match="outside the instruction root"):
        discover(root, home)


def test_links_inside_code_and_non_markdown_links_are_not_directives(tmp_path, home):
    root = project(tmp_path)
    write(root / "AGENTS.md", "Example: `[x](missing.md)`\n\n```\n[y](also-missing.md)\n```\n\n"
                              "See [the code](src/app.py) and [docs](https://example.com/page).")
    found = discover(root, home)
    assert found.files == ("AGENTS.md",)


# --- bounds ---------------------------------------------------------------------------------------------------------

def test_the_total_size_limit_refuses_the_set_without_truncating_it(tmp_path, home, monkeypatch):
    monkeypatch.setattr(instructions, "MAX_INSTRUCTION_BYTES", 1000)
    root = project(tmp_path)
    write(root / "AGENTS.md", "[A](a.md) [B](b.md)")
    write(root / "a.md", "a" * 600)
    write(root / "b.md", "b" * 600)
    with pytest.raises(instructions.InstructionError, match="exceed the limit of 1000 bytes"):
        discover(root, home)


def test_a_single_oversized_document_is_refused(tmp_path, home, monkeypatch):
    monkeypatch.setattr(instructions, "MAX_INSTRUCTION_BYTES", 100)
    root = project(tmp_path)
    write(root / "AGENTS.md", "x" * 101)
    with pytest.raises(instructions.InstructionError, match="larger than the instruction limit"):
        discover(root, home)


def test_an_unreadable_or_non_utf8_document_is_reported(tmp_path, home):
    root = project(tmp_path)
    (root / "AGENTS.md").write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(instructions.InstructionError, match="not UTF-8"):
        discover(root, home)


# --- run integration: prompt position, once, named agents, refusal before any request -------------------------------

def run_goal(goal, **kwargs):
    event_loop = asyncio.new_event_loop()
    try:
        return event_loop.run_until_complete(
            loop.agent_loop_async(goal, SimpleNamespace(get_openai_tools=lambda: []), event_loop, **kwargs))
    finally:
        event_loop.close()


@pytest.fixture
def run_env(tmp_path, home, monkeypatch, tmp_sessions):
    root = project(tmp_path)
    monkeypatch.chdir(root)
    monkeypatch.setattr(loop, "load_project_guidelines", lambda: instructions.discover(Path.cwd().resolve(), home))
    return root


def test_instructions_are_in_the_first_request_once_and_after_a_named_agent_override(run_env, fake_post,
                                                                                    streamed_response):
    write(run_env / "AGENTS.md", "- [Workflow](WO.md)")
    write(run_env / "WO.md", "UNIQUE-WORKFLOW-RULE")
    post = fake_post([streamed_response(finish_reason="stop", content="done")])
    run_goal("goal", system_prompt_override="AGENT-PERSONA-PROMPT")
    system = post.payloads[0]["messages"][0]["content"]
    assert system.startswith("AGENT-PERSONA-PROMPT")
    assert system.index("AGENT-PERSONA-PROMPT") < system.index("UNIQUE-WORKFLOW-RULE")
    assert system.count("UNIQUE-WORKFLOW-RULE") == 1
    assert "grant no tools, approvals or paths" in system


def test_a_missing_linked_document_refuses_the_run_before_any_model_request(run_env, fake_post, streamed_response,
                                                                            capsys):
    write(run_env / "AGENTS.md", "[Workflow](WO.md)")  # WO.md is absent
    post = fake_post([streamed_response(finish_reason="stop", content="never")])
    outcome = run_goal("goal")
    assert post.payloads == [], "no model request may be made with incomplete instructions"
    assert outcome.status == runs.RunStatus.NEEDS_ATTENTION.value
    assert "WO.md" in capsys.readouterr().out


def test_discovery_leaves_the_execution_workspace_and_its_tool_permissions_unchanged(run_env, fake_post,
                                                                                   streamed_response, tmp_path):
    nested = run_env / "tmp" / "test"
    nested.mkdir(parents=True)
    write(run_env / "AGENTS.md", "Root rules.")
    tools_before = [t["function"]["name"] for t in get_local_tools()]
    fake_post([streamed_response(finish_reason="stop", content="done")])
    os.chdir(nested)
    try:
        run_goal("goal")
        assert Path.cwd() == nested.resolve()
        refused = read_file_safe("../../AGENTS.md")  # the parent instruction file is outside the workspace
        assert refused["isError"] is True and refused["errorCategory"] == "permission"
    finally:
        os.chdir(run_env)
    assert [t["function"]["name"] for t in get_local_tools()] == tools_before
