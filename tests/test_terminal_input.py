"""Interactive input through a real pseudo-terminal: pastes, typed keys, permission answers, cancellation, and working
directory. Each test runs the real terminal_input module in a child process whose standard input is a pty. Only the
terminal is simulated, by writing the bytes a terminal would send. No model, no network."""

import fcntl
import json
import os
import pty
import select
import signal
import subprocess
import sys
import termios
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PASTE_START, PASTE_END = "\x1b[200~", "\x1b[201~"

GOAL = f"""
import json, sys
sys.path.insert(0, {str(REPO)!r})
from src import terminal_input as t
try:
    goal = t.read_goal('GOAL> ')
except KeyboardInterrupt:
    print('\\nINTERRUPTED', flush=True)
except EOFError:
    print('\\nEOF', flush=True)
else:
    print('\\nGOAL ' + json.dumps(goal), flush=True)
"""

CHOICE = f"""
import sys
sys.path.insert(0, {str(REPO)!r})
from src import terminal_input as t
answer = t.read_choice('ALLOW? (1/2/3): ', ['1', '2', '3'])
print('\\nCHOICE ' + answer, flush=True)
"""

GOAL_THEN_CHOICE = f"""
import json, sys
sys.path.insert(0, {str(REPO)!r})
from src import terminal_input as t
goal = t.read_goal('GOAL> ')
print('\\nGOAL ' + json.dumps(goal), flush=True)
answer = t.read_choice('ALLOW? (1/2/3): ', ['1', '2', '3'])
print('\\nCHOICE ' + answer, flush=True)
"""


def _become_terminal_session() -> None:
    """Make the pty the child's controlling terminal, as a real terminal is. Only then does Ctrl-C become SIGINT.

    SIGINT is reset to its default before exec. A background job (a shell started the pytest run with `&`) inherits
    SIG_IGN for SIGINT; ignored dispositions survive exec, so without this reset Python would never install its interrupt
    handler, and a Ctrl-C would be dropped silently. An interactive terminal's foreground program has the default."""
    os.setsid()
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    fcntl.ioctl(0, termios.TIOCSCTTY, 0)


class Terminal:
    """A child process attached to a pty. Output is collected; input is written as a terminal would send it."""

    def __init__(self, script: str, pre_input: bytes = b"", cwd=None):
        self.master, slave = pty.openpty()
        if pre_input:
            os.write(self.master, pre_input)  # queued before the program starts
        self.proc = subprocess.Popen([sys.executable, "-c", script], stdin=slave, stdout=slave, stderr=slave,
                                     cwd=cwd, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), close_fds=True,
                                     preexec_fn=_become_terminal_session)
        os.close(slave)
        self.transcript = b""
        self.mark = 0

    def _pump(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            ready, _, _ = select.select([self.master], [], [], 0.05)
            if ready:
                try:
                    self.transcript += os.read(self.master, 65536)
                except OSError:
                    return

    def settle(self, seconds: float = 0.4) -> None:
        self._pump(seconds)

    def expect(self, text: str, timeout: float = 15.0) -> str:
        """Wait until text appears after the last expectation; return the text seen since then. The mark moves only
        to the end of the match, so output that arrived in the same read is still available to the next expectation."""
        end = time.monotonic() + timeout
        while True:
            seen = self.transcript[self.mark:].decode("utf-8", "replace")
            if text in seen:
                upto = seen.index(text) + len(text)
                self.mark += len(seen[:upto].encode("utf-8"))
                return seen
            assert time.monotonic() < end, f"never saw {text!r}; transcript tail: {seen[-300:]!r}"
            self._pump(0.05)

    def absent(self, text: str, seconds: float = 0.6) -> None:
        self._pump(seconds)
        seen = self.transcript[self.mark:].decode("utf-8", "replace")
        assert text not in seen, f"unexpected {text!r} in {seen[-300:]!r}"

    def send(self, data) -> None:
        raw = data.encode("utf-8") if isinstance(data, str) else data
        while raw:
            written = os.write(self.master, raw)
            raw = raw[written:]

    def key(self, text: str) -> None:
        """Keys typed one at a time, each a separate read, as a person types."""
        for character in text:
            self.send(character)
            self._pump(0.03)

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait(timeout=10)
        os.close(self.master)


def _goal_value(seen: str):
    line = next(line for line in seen.splitlines() if line.startswith("GOAL "))
    return json.loads(line[len("GOAL "):])


@pytest.fixture
def term():
    made = []

    def make(script, **kwargs):
        terminal = Terminal(script, **kwargs)
        made.append(terminal)
        return terminal

    yield make
    for terminal in made:
        terminal.close()


MARKDOWN = "\n".join(
    [f"## {n}. Section {n}\n\nText for section {n}: café, naïve, 日本語, 🚀.\n\n- item {n}a\n- item {n}b\n"
     for n in range(1, 151)])


# --- 1. small single-line input, and normal editing ---------------------------------------------------------------

def test_single_line_input_is_submitted_by_enter_and_editing_keys_work(term):
    t = term(GOAL)
    t.expect("GOAL> ")
    t.key("fix the bugz")
    t.send(b"\x7f")
    # Wait until the program has read and echoed the Backspace. Only then is Enter sent, so it arrives as its own read.
    # Written back to back, the two bytes can share one read, and a newline in a multi-byte read is pasted text by design.
    t.expect("\b \b")
    t.send("\r")
    assert _goal_value(t.expect("GOAL ")) == "fix the bug"


# --- 2. a large multiline Markdown paste is one input operation -------------------------------------------------------

def test_large_multiline_markdown_paste_is_one_operation_and_preserved_exactly(term):
    t = term(GOAL)
    t.expect("GOAL> ")
    started = time.monotonic()
    t.send(PASTE_START + MARKDOWN + PASTE_END)
    t.absent("GOAL ", seconds=0.8)  # no premature submission at the first of 450 newlines
    t.send("\r")
    assert _goal_value(t.expect("GOAL ")) == MARKDOWN
    assert time.monotonic() - started < 10, "a large paste must stay responsive"


# --- 3. code fences, Unicode, tabs ----------------------------------------------------------------------------------

def test_code_fences_unicode_and_tabs_are_preserved(term):
    text = "Run:\n```python\n\tprint('héllo 🚀 日本')\n```\n\nÉcole — “quotes” ∑ ñ\n"
    t = term(GOAL)
    t.expect("GOAL> ")
    t.send(PASTE_START + text + PASTE_END)
    t.settle(0.3)
    t.send("\r")
    assert _goal_value(t.expect("GOAL ")) == text


# --- 4. paste followed by normal typing --------------------------------------------------------------------------------

def test_typing_after_a_paste_continues_the_same_goal(term):
    t = term(GOAL)
    t.expect("GOAL> ")
    t.send(PASTE_START + "line one\nline two" + PASTE_END)
    t.settle(0.3)
    t.key(" and then")
    t.send("\r")
    assert _goal_value(t.expect("GOAL ")) == "line one\nline two and then"


# --- 5. a paste immediately preceding a permission request ------------------------------------------------------------

def test_paste_before_a_permission_request_cannot_answer_it(term):
    t = term(GOAL_THEN_CHOICE)
    t.expect("GOAL> ")
    t.send(PASTE_START + "## 1. Plan\n## 2. Resume\n3\n" + PASTE_END)
    t.settle(0.3)
    t.send("\r")
    assert _goal_value(t.expect("GOAL ")) == "## 1. Plan\n## 2. Resume\n3\n"
    t.expect("ALLOW?")
    t.absent("CHOICE", seconds=0.8)  # the pasted 2 and 3 were goal text, not answers
    t.key("1")
    assert "CHOICE 1" in t.expect("CHOICE ")


# --- 6. pasted digits never answer a permission prompt --------------------------------------------------------------

def test_pasted_digits_are_not_an_answer_to_a_permission_prompt(term):
    t = term(CHOICE)
    t.expect("ALLOW?")
    t.send(PASTE_START + "1\n2\n3\n" + PASTE_END)
    t.expect("not an answer")
    t.absent("CHOICE", seconds=0.6)
    t.key("3")
    assert "CHOICE 3" in t.expect("CHOICE ")


def test_a_multi_byte_burst_without_paste_markers_is_not_an_answer(term):
    t = term(CHOICE)
    t.expect("ALLOW?")
    t.send("2\n")  # several bytes in one read: not a keypress, even without bracketed paste
    t.expect("not an answer")
    t.absent("CHOICE", seconds=0.5)
    t.key("1")
    assert "CHOICE 1" in t.expect("CHOICE ")


def test_an_arrow_key_is_not_an_answer(term):
    t = term(CHOICE)
    t.expect("ALLOW?")
    t.send("\x1b[A")
    t.absent("CHOICE", seconds=0.5)
    t.key("2")
    assert "CHOICE 2" in t.expect("CHOICE ")


# --- 7. residual buffered input when the permission prompt appears ---------------------------------------------------

def test_residual_input_queued_before_the_prompt_is_discarded_not_answered(term):
    # A single leftover byte with no newline: the multi-byte rule cannot catch it, so only the discard protects the prompt
    t = term(CHOICE, pre_input=b"2")
    t.expect("ALLOW?")
    t.absent("CHOICE", seconds=0.8)
    t.key("2")
    assert "CHOICE 2" in t.expect("CHOICE ")


def test_a_queued_multi_key_leftover_is_not_an_answer(term):
    t = term(CHOICE, pre_input=b"2\n2\n1\n")
    t.expect("ALLOW?")
    t.absent("CHOICE", seconds=0.8)
    t.key("3")
    assert "CHOICE 3" in t.expect("CHOICE ")


# --- 8. cancellation during input ---------------------------------------------------------------------------------------

def test_ctrl_c_during_goal_input_interrupts_even_inside_an_open_paste(term):
    t = term(GOAL)
    t.expect("GOAL> ")
    t.send(PASTE_START + "half a paste that never closes")
    t.settle(0.2)
    t.send(b"\x03")
    assert "INTERRUPTED" in t.expect("INTERRUPTED")


def test_ctrl_c_at_a_permission_prompt_denies_immediately(term):
    t = term(CHOICE)
    t.expect("ALLOW?")
    t.send(b"\x03")
    assert "CHOICE 3" in t.expect("CHOICE ")


def test_ctrl_d_on_an_empty_goal_ends_input(term):
    t = term(GOAL)
    t.expect("GOAL> ")
    t.send(b"\x04")
    assert "EOF" in t.expect("EOF")


# --- non-terminal input keeps the previous behaviour --------------------------------------------------------------------

def test_piped_input_is_read_as_before(tmp_path):
    script = GOAL
    proc = subprocess.run([sys.executable, "-c", script], input=b"piped goal\n", capture_output=True,
                          cwd=tmp_path, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), timeout=30)
    assert _goal_value(proc.stdout.decode()) == "piped goal"


# --- 9. running from a nested working directory -----------------------------------------------------------------------

def test_the_workspace_is_the_working_directory_not_the_script_location(tmp_path, monkeypatch):
    from src import loop
    from src.tools.files import read_file_safe

    workspace = tmp_path / "project" / "nested"
    workspace.mkdir(parents=True)
    (workspace / "AGENTS.md").write_text("nested guidelines", encoding="utf-8")
    (workspace / "notes.txt").write_text("inside", encoding="utf-8")
    (tmp_path / "project" / "outside.txt").write_text("outside", encoding="utf-8")
    monkeypatch.chdir(workspace)

    assert "nested guidelines" in loop.load_project_guidelines().text
    assert read_file_safe("notes.txt")["content"] == "inside"
    refused = read_file_safe("../outside.txt")
    assert refused["isError"] is True and refused["errorCategory"] == "permission"
    refused_repo = read_file_safe(str(REPO / "README.md"))
    assert refused_repo["isError"] is True


def test_a_terminal_started_in_a_nested_directory_uses_it_as_the_workspace(tmp_path):
    nested = tmp_path / "tmp" / "test"
    nested.mkdir(parents=True)
    (nested / "AGENTS.md").write_text("nested", encoding="utf-8")
    script = f"""
import os, sys
sys.path.insert(0, {str(REPO)!r})
from src import loop
print('CWD ' + os.getcwd())
guide = loop.load_project_guidelines()
print('GUIDE ' + (guide.text if guide else 'None'))
"""
    proc = subprocess.run([sys.executable, "-c", script], cwd=nested, capture_output=True, text=True, timeout=30,
                          env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert f"CWD {nested.resolve()}" in proc.stdout and "GUIDE --- AGENTS.md ---\nnested" in proc.stdout


# --- the REPL itself: a pasted multi-line goal is not submitted, and Ctrl-C ends the session -----------------------------

def test_the_repl_keeps_a_multiline_paste_unsubmitted_and_ctrl_c_exits_without_a_model_request(tmp_path):
    """Drives agent_loop.py repl from a nested directory. The goal is never submitted, so no model request is made."""
    nested = tmp_path / "tmp" / "test"
    nested.mkdir(parents=True)
    t = Terminal.__new__(Terminal)
    t.master, slave = pty.openpty()
    t.proc = subprocess.Popen([sys.executable, str(REPO / "agent_loop.py"), "repl"], stdin=slave, stdout=slave,
                              stderr=slave, cwd=nested, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                              close_fds=True, preexec_fn=_become_terminal_session)
    os.close(slave)
    t.transcript, t.mark = b"", 0
    try:
        t.expect("Enter goal")
        t.send(PASTE_START + "## 1. Plan\n## 2. Resume Context\n- item 3\n" + PASTE_END)
        t.settle(0.8)
        t.absent("Starting agentic loop", seconds=0.5)  # nothing was submitted
        t.send(b"\x03")  # Ctrl-C at the goal prompt
        t.expect("Exiting")
        assert t.proc.wait(timeout=20) == 0
    finally:
        if t.proc.poll() is None:
            t.proc.kill()
        t.proc.wait(timeout=10)  # reap it, so no unwaited child is left behind
        os.close(t.master)
