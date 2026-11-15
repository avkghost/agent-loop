"""Terminal input for the REPL goal prompt and the permission prompts.

Why this exists. The REPL read its goal with input(), which is line-buffered: a multi-line paste was submitted at its
first newline, and the rest stayed in the terminal's input queue. The permission prompt then read single characters from
that same queue through Python's buffered sys.stdin, so pasted digits were taken as answers. A digit inside a pasted
heading ("## 2. Resume") granted "Yes, this session".

What it does instead.
  * Bytes are read from the file descriptor directly. Python's text buffer cannot hide bytes from the kernel queue.
  * Bracketed paste mode (ESC[?2004h) makes the terminal wrap a paste in ESC[200~ ... ESC[201~. A paste is one unit of
    text, newlines included. It never submits a goal and never answers a prompt.
  * A read that delivers several bytes at once is not treated as a keypress. A newline inside such a read is text, so a
    paste from a terminal without bracketed paste does not submit early either.
  * A permission prompt first discards everything already queued, then accepts only one deliberate key. Pasted text,
    multi-byte reads, and anything that is not an allowed key are ignored, and the prompt waits.
  * Ctrl-C raises KeyboardInterrupt in the goal prompt (as input() did) and denies in the permission prompt.

Without a terminal (pipes, redirected input, Windows) both functions fall back to input(), as before.
"""

import codecs
import contextlib
import os
import sys

try:
    import fcntl
    import termios
    import tty
except ImportError:  # pragma: no cover - Windows
    fcntl = termios = tty = None

PASTE_ON = "\x1b[?2004h"
PASTE_OFF = "\x1b[?2004l"
PASTE_START = b"\x1b[200~"
PASTE_END = b"\x1b[201~"
_CHUNK = 65536


def _stdin_fd():
    """The terminal's file descriptor, or None when standard input is not an interactive terminal."""
    if termios is None:
        return None
    try:
        fd = sys.stdin.fileno()
    except (AttributeError, OSError, ValueError):
        return None
    return fd if os.isatty(fd) else None


def _partial_suffix(data: bytes, marker: bytes) -> int:
    """Length of the longest suffix of data that is a proper prefix of marker (it may complete on the next read)."""
    for length in range(min(len(marker) - 1, len(data)), 0, -1):
        if data.endswith(marker[:length]):
            return length
    return 0


def _csi_end(data: bytes) -> int:
    """Index just past a control sequence that starts with ESC [, or -1 while it is incomplete."""
    for index in range(2, len(data)):
        if 0x40 <= data[index] <= 0x7E:
            return index + 1
    return -1


class _Stream:
    """Terminal bytes turned into events. Keeps an incomplete escape sequence until its remainder arrives."""

    def __init__(self, fd: int):
        self.fd = fd
        self.buffer = b""
        self.text = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.paste = None  # bytes collected while inside a bracketed paste; None outside one

    def read(self):
        """Block for terminal input. Returns (bytes_in_this_read, events). Raises EOFError at end of input."""
        data = os.read(self.fd, _CHUNK)
        if not data:
            raise EOFError
        self.buffer += data
        return len(data), self._events(multi=len(data) > 1)

    def _events(self, multi: bool):
        events = []
        while self.buffer:
            if self.paste is not None:
                end = self.buffer.find(PASTE_END)
                interrupt = self.buffer.find(b"\x03")
                if interrupt >= 0 and (end < 0 or interrupt < end):
                    # A Ctrl-C is never part of a paste: it cancels, even if the paste was never closed
                    self.buffer = self.buffer[interrupt + 1:]
                    self.paste = None
                    events.append(("char", "\x03", False))
                    continue
                if end < 0:
                    keep = _partial_suffix(self.buffer, PASTE_END)
                    self.paste.append(self.buffer[:len(self.buffer) - keep])
                    self.buffer = self.buffer[len(self.buffer) - keep:]
                    break
                self.paste.append(self.buffer[:end])
                self.buffer = self.buffer[end + len(PASTE_END):]
                events.append(("paste", b"".join(self.paste).decode("utf-8", errors="replace")))
                self.paste = None
                continue
            if self.buffer.startswith(PASTE_START):
                self.buffer = self.buffer[len(PASTE_START):]
                self.paste = []
                continue
            if PASTE_START.startswith(self.buffer):
                break  # could still become the start of a paste
            if self.buffer.startswith(b"\x1b["):
                end = _csi_end(self.buffer)
                if end < 0:
                    break
                self.buffer = self.buffer[end:]  # arrow keys and other control sequences carry no text
                continue
            if self.buffer.startswith(b"\x1b"):
                self.buffer = self.buffer[1:]
                continue
            cut = self.buffer.find(b"\x1b")
            cut = len(self.buffer) if cut < 0 else cut
            text = self.text.decode(self.buffer[:cut])
            self.buffer = self.buffer[cut:]
            events.extend(("char", character, multi) for character in text)
        return events


@contextlib.contextmanager
def _terminal(fd: int):
    """Character-at-a-time input with no local echo; signals stay enabled. Bracketed paste is on while it is open."""
    saved = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    sys.stdout.write(PASTE_ON)
    sys.stdout.flush()
    try:
        yield
    finally:
        sys.stdout.write(PASTE_OFF)
        sys.stdout.flush()
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def _queued_bytes(fd: int) -> int:
    buf = fcntl.ioctl(fd, termios.FIONREAD, b"\x00\x00\x00\x00")
    return int.from_bytes(buf, "little")


def _discard_queued(fd: int) -> int:
    """Throw away everything the terminal holds, including an incomplete line that FIONREAD does not count.
    Returns the number of readable bytes that were discarded (for the notice only)."""
    count = _queued_bytes(fd)
    termios.tcflush(fd, termios.TCIFLUSH)
    return count


def read_goal(prompt: str) -> str:
    """One goal. A paste is taken whole, newlines included; the goal is submitted only by a deliberate Enter."""
    fd = _stdin_fd()
    if fd is None:
        return input(prompt)
    print(prompt, end="", flush=True)
    parts = []
    with _terminal(fd):
        stream = _Stream(fd)
        while True:
            _, events = stream.read()
            for event in events:
                if event[0] == "paste":
                    text = event[1]
                    parts.append(text)
                    lines = text.count("\n") + 1
                    sys.stdout.write(f"[pasted {len(text)} characters, {lines} lines]")
                    sys.stdout.flush()
                    continue
                _, character, multi = event
                if character in "\r\n":
                    if multi:  # a newline inside a multi-byte read is pasted text, never a submission
                        parts.append("\n")
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                        continue
                    discarded = _discard_queued(fd)
                    if discarded:
                        print(f"\n(discarded {discarded} byte(s) typed ahead of the submission)")
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    return "".join(parts)
                if character == "\x03":
                    raise KeyboardInterrupt
                if character == "\x04":  # Ctrl-D: end of input when nothing has been typed, as input() does
                    if not "".join(parts):
                        raise EOFError
                    continue
                if character in "\x7f\x08":
                    if parts and parts[-1]:
                        parts[-1] = parts[-1][:-1]
                        if not parts[-1]:
                            parts.pop()
                        sys.stdout.write("\b \b")
                        sys.stdout.flush()
                    continue
                if character < " ":
                    continue
                parts.append(character)
                sys.stdout.write(character)
                sys.stdout.flush()


def read_choice(prompt: str, valid_keys: list) -> str:
    """A permission answer: one deliberate key from valid_keys. Ctrl-C denies ("3"). Nothing else is an answer."""
    fd = _stdin_fd()
    if fd is None:
        return input(prompt).strip() or "3"
    try:  # the prompt is inside the protected block: a Ctrl-C that lands while it is drawn still denies
        print(prompt, end="", flush=True)
        with _terminal(fd):
            # Input queued before this prompt (typed ahead, or left over from a paste) is not an answer to it.
            _discard_queued(fd)
            stream = _Stream(fd)
            while True:
                count, events = stream.read()
                for event in events:
                    if event[0] == "paste" or count != 1:
                        _ignored(valid_keys)
                        break
                    character = event[1]
                    if character == "\x03":
                        raise KeyboardInterrupt
                    if character in valid_keys:
                        print(character)
                        return character
                    _ignored(valid_keys)
                    break
    except KeyboardInterrupt:
        # Ctrl-C denies the request; it never aborts the run from inside a permission prompt
        print("\n❌ Interrupted - Denying.")
        return "3"


def _ignored(valid_keys: list) -> None:
    keys = ", ".join(valid_keys[:-1]) + " or " + valid_keys[-1]
    sys.stdout.write(f"\n(pasted or multi-key input is not an answer; press {keys})\n")
    sys.stdout.flush()
