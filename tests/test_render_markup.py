"""Tool output is data, not Rich markup. Bracketed text from files, commands and the model must display as written
and must never end a run; the text sent to the model is never changed for display.

Offline only. The console is replaced by an in-memory one, and the model is scripted."""

import io
import json
import re

import pytest
from rich.console import Console

from src import envelope, render
from src.tools import files
from tests.conftest import make_streamed_response

RUST_ATTRIBUTES = (
    "#[derive(Debug, Clone, Serialize, Deserialize)]\n"
    "struct User {}\n"
    "#[tokio::main]\n"
    "async fn main() {}\n"
    "#[utoipa::path(get, path = \"/users\", responses((status = 200, description = \"ok\")))]\n"
    "#[cfg(test)]\n"
)
CLOSING_STYLE = "// [/x] and [bold] and [a](b)\nfn f() {}\n"


@pytest.fixture
def screen(monkeypatch):
    """The renderer's console writes to a buffer, so the text a person would see can be compared exactly."""
    buffer = io.StringIO()
    monkeypatch.setattr(render, "console", Console(file=buffer, width=120, color_system=None, force_terminal=False))
    return buffer


def observation(content, is_error=False, error_category=None, is_retryable=None):
    env = envelope.error_validation(content) if is_error else envelope.success(content)
    if error_category:
        env["errorCategory"] = error_category
    if is_retryable is not None:
        env["isRetryable"] = is_retryable
    return envelope.serialize_for_model(env)


# --- tool output: displayed exactly -------------------------------------------------------------------------------

def test_rust_attributes_are_displayed_whole_not_reduced_to_a_bare_hash(screen):
    render.print_observation(observation(RUST_ATTRIBUTES))
    shown = screen.getvalue()
    assert "#[derive(Debug, Clone, Serialize, Deserialize)]" in shown
    assert "#[tokio::main]" in shown
    assert '#[utoipa::path(get, path = "/users", responses((status = 200, description = "ok")))]' in shown
    assert "#[cfg(test)]" in shown


def test_closing_style_and_link_like_brackets_do_not_raise_and_are_shown(screen):
    render.print_observation(observation(CLOSING_STYLE))  # used to raise MarkupError and end the run
    shown = screen.getvalue()
    assert "[/x]" in shown and "[bold]" in shown and "[a](b)" in shown


def test_error_metadata_is_still_shown_after_the_content(screen):
    render.print_observation(observation("[/x] failed", is_error=True, error_category="validation", is_retryable=False))
    shown = screen.getvalue()
    assert "[/x] failed" in shown
    assert "errorCategory: validation" in shown and "isRetryable: False" in shown


def test_plain_text_fallback_shows_brackets(screen):
    render.print_observation("plain #[derive(Debug)] text [/x]")
    assert "plain #[derive(Debug)] text [/x]" in screen.getvalue()


def test_tool_call_name_and_arguments_are_shown_as_written(screen):
    render.print_tool_call("read_file", {"path": "src/#[derive(Debug)].rs"})
    render.print_tool_call("run_command", {"command": "echo [/x]"})  # used to raise MarkupError
    shown = screen.getvalue()
    assert "#[derive(Debug)].rs" in shown
    assert "echo [/x]" in shown


def test_permission_identifier_is_shown_as_written(screen):
    render.print_permission_request("command", "grep '#[derive(Debug)]' src/lib.rs")
    render.print_permission_request("command", "echo [/x]")  # used to raise MarkupError
    shown = screen.getvalue()
    assert "grep '#[derive(Debug)]' src/lib.rs" in shown
    assert "echo [/x]" in shown


# --- the model-facing text is untouched --------------------------------------------------------------------------

def test_the_observation_sent_to_the_model_keeps_the_source_bytes_exactly():
    stored = observation(RUST_ATTRIBUTES)
    assert json.loads(stored)["content"] == RUST_ATTRIBUTES
    assert "#[derive(Debug, Clone, Serialize, Deserialize)]" in stored


def test_read_file_content_is_the_source_text(tmp_path, monkeypatch):
    (tmp_path / "lib.rs").write_text(RUST_ATTRIBUTES + CLOSING_STYLE, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert files.read_file_safe("lib.rs")["content"] == RUST_ATTRIBUTES + CLOSING_STYLE


# --- existing formatting is unchanged ----------------------------------------------------------------------------

def test_assistant_markdown_still_formats_and_keeps_bracketed_code(screen):
    render.print_assistant_message("**bold** and `#[derive(Debug)]`\n\n```rust\n#[cfg(test)]\n```")
    shown = screen.getvalue()
    assert "**" not in shown and "bold" in shown
    assert "#[derive(Debug)]" in shown and "#[cfg(test)]" in shown


def test_markup_styling_of_render_helpers_is_unchanged(screen):
    render.print_iteration_header(2, 100)
    render.print_guardrail_stop()
    shown = screen.getvalue()
    assert "ITERATION 2/100" in shown and "Guardrail: No progress detected" in shown
    assert "[cyan]" not in shown and "[/" not in shown  # markup is applied, not printed


# --- end to end: a file with closing-style text does not end the run --------------------------------------------

def test_a_run_reading_closing_style_text_completes_and_sends_the_next_request(run_loop, monkeypatch, tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "closing.rs").write_text(CLOSING_STYLE, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    responses = [
        make_streamed_response(finish_reason="tool_calls", tool_calls=[
            {"id": "c1", "name": "read_file", "arguments": json.dumps({"path": "src/closing.rs"})}]),
        make_streamed_response(finish_reason="stop", content="done"),
    ]
    run_loop(responses, dispatch_fn=lambda tool, inp, *a, **k: files.read_file_safe(inp["path"]))
    assert len(run_loop.last_post.payloads) == 2, "the second request must be sent after the tool result"
    tool_message = next(m for m in run_loop.last_post.payloads[1]["messages"] if m.get("role") == "tool")
    assert json.loads(tool_message["content"])["content"] == CLOSING_STYLE


# --- every dynamic string printed through a console markup path is literal -----------------------------------------

PAYLOADS = ["[/x] tail", "[bold]not bold[/bold]", "#[derive(Debug, Clone)] struct A", "[cyan]c", "a \\[escaped] b"]

DYNAMIC_OUTPUT = {
    "banner goal": lambda p: render.print_banner(p, 3),
    "llm error": render.print_llm_error,
    "finish reason": render.print_finish_reason_warning,
    "permission identifier": lambda p: render.print_permission_request("command", p),
    "permission result label": render.print_permission_result,
    "completion note": lambda p: render.print_completion(note=p),
}


@pytest.mark.parametrize("payload", PAYLOADS)
@pytest.mark.parametrize("site", sorted(DYNAMIC_OUTPUT))
def test_dynamic_text_is_printed_literally_and_never_raises(screen, site, payload):
    DYNAMIC_OUTPUT[site](payload)  # used to raise MarkupError for "[/x]" in four of these sites
    assert payload.replace("\\", "") in screen.getvalue().replace("\\", "")


def test_the_numeric_and_constant_notices_are_unchanged(screen):
    render.print_retry_notice(2, 3, 4.0)
    render.print_compaction_notice(100, 40)
    render.print_empty_response()
    shown = screen.getvalue()
    assert "Retry 2/3, waiting 4s" in shown
    assert "Context size reduced to 40 tokens" in shown
    assert "Empty response from LM Studio" in shown


@pytest.fixture
def styled_screen(monkeypatch):
    """A colour-capable console, so intentional styling is visible as terminal escape codes."""
    buffer = io.StringIO()
    monkeypatch.setattr(render, "console", Console(file=buffer, width=120, color_system="standard", force_terminal=True))
    return buffer


def _visible(text):
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def test_intentional_styling_still_applies_around_literal_text(styled_screen):
    render.print_llm_error("[/x] boom")
    shown = styled_screen.getvalue()
    assert "\x1b[31m" in shown, "the error line keeps its red style"
    assert "[/x] boom" in _visible(shown), "the dynamic part is literal"
    styled_screen.seek(0); styled_screen.truncate()
    render.print_banner("[bold]goal[/bold]", 2)
    banner = styled_screen.getvalue()
    assert re.search(r"\x1b\[(?:[0-9;]*;)?1(?:;[0-9;]*)?m", banner), "the banner keeps its bold style"
    assert "[bold]goal[/bold]" in _visible(banner)
