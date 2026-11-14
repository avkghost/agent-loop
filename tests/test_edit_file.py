"""edit_file: exact-match replacement with the refusal, containment and atomicity guarantees of the file tools.

Every refusal is checked by its bytes: a refused edit must leave the file exactly as it was, and must leave no
temporary file behind.
"""

import asyncio
import os
import stat
from pathlib import Path
from unittest import mock

import pytest

from src import dispatch
from src.authorization import SessionAuthorization
from src.tools import files, get_local_tools, local_registry
from src.tools.registry import FILE


@pytest.fixture
def ws(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _leftover_temporaries(directory: Path):
    return sorted(p.name for p in directory.glob(".*.tmp"))


def _edit(path, old, new):
    return files.edit_file_safe(path, old, new)


def test_single_occurrence_is_replaced_and_the_rest_is_kept(ws):
    target = ws / "app.py"
    target.write_bytes(b"def a():\n    return 1\n\ndef b():\n    return 3\n")
    result = _edit("app.py", "    return 1\n", "    return 2\n")
    assert result["isError"] is False
    assert target.read_bytes() == b"def a():\n    return 2\n\ndef b():\n    return 3\n"
    assert "at line 2" in result["content"]
    assert len(result["content"]) < 200  # the result is bounded: it never echoes the file


def test_multiline_old_string_can_be_replaced_and_deleted(ws):
    target = ws / "notes.txt"
    target.write_bytes(b"keep\nremove me\nand me\nkeep too\n")
    assert _edit("notes.txt", "remove me\nand me\n", "")["isError"] is False
    assert target.read_bytes() == b"keep\nkeep too\n"


def test_missing_match_is_refused_and_changes_nothing(ws):
    target = ws / "f.txt"
    original = b"alpha\nbeta\n"
    target.write_bytes(original)
    result = _edit("f.txt", "gamma", "delta")
    assert result["isError"] is True
    assert result["errorCategory"] == "validation" and result["isRetryable"] is True
    assert "not found" in result["content"]
    assert target.read_bytes() == original
    assert _leftover_temporaries(ws) == []


def test_multiple_matches_are_refused_and_changes_nothing(ws):
    target = ws / "f.txt"
    original = b"x = 1\nx = 1\n"
    target.write_bytes(original)
    result = _edit("f.txt", "x = 1\n", "x = 2\n")
    assert result["isError"] is True and "more than one location" in result["content"]
    assert target.read_bytes() == original


def test_overlapping_occurrences_count_as_ambiguous(ws):
    target = ws / "f.txt"
    target.write_bytes(b"aaa")
    result = _edit("f.txt", "aa", "b")  # "aa" occurs at offsets 0 and 1
    assert result["isError"] is True
    assert target.read_bytes() == b"aaa"


@pytest.mark.parametrize("old, new, message", [
    ("", "x", "non-empty"),
    ("same", "same", "identical"),
    (None, "x", "non-empty"),
    ("a", None, "must be a string"),
])
def test_invalid_string_arguments_are_refused(ws, old, new, message):
    target = ws / "f.txt"
    target.write_bytes(b"same\n")
    result = _edit("f.txt", old, new)
    assert result["isError"] is True and message in result["content"]
    assert target.read_bytes() == b"same\n"


def test_oversized_strings_are_refused(ws):
    target = ws / "f.txt"
    target.write_bytes(b"a\n")
    too_long = "a" * (files.EDIT_MAX_STRING_CHARS + 1)
    result = _edit("f.txt", too_long, "b")
    assert result["isError"] is True and "limited to" in result["content"]
    assert target.read_bytes() == b"a\n"


def test_stale_content_is_refused_and_the_other_writers_bytes_survive(ws, monkeypatch):
    target = ws / "f.txt"
    target.write_bytes(b"one\ntwo\n")
    real_read_bytes = Path.read_bytes
    reads = {"count": 0}

    def racing_read(self, *args, **kwargs):
        # The second read is the stale-edit check. Another writer changes the file between the two reads.
        if self.name == "f.txt":
            reads["count"] += 1
            if reads["count"] == 2:
                self.write_bytes(b"one\nTWO-CHANGED-ELSEWHERE\n")
        return real_read_bytes(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", racing_read)
    result = _edit("f.txt", "two\n", "TWO\n")
    monkeypatch.setattr(Path, "read_bytes", real_read_bytes)

    assert reads["count"] == 2
    assert result["isError"] is True and "changed while the edit was being prepared" in result["content"]
    assert target.read_bytes() == b"one\nTWO-CHANGED-ELSEWHERE\n"
    assert _leftover_temporaries(ws) == []


def test_fsync_failure_leaves_the_original_and_no_temporary(ws, monkeypatch):
    target = ws / "f.txt"
    original = b"before\n"
    target.write_bytes(original)

    def failing_fsync(fd):
        raise OSError(5, "simulated disk failure")

    with mock.patch.object(os, "fsync", failing_fsync):
        result = _edit("f.txt", "before", "after")
    assert result["isError"] is True and "existing file was not modified" in result["content"]
    assert target.read_bytes() == original
    assert _leftover_temporaries(ws) == []


def test_replace_failure_before_the_swap_is_definitive(ws, monkeypatch):
    target = ws / "f.txt"
    original = b"before\n"
    target.write_bytes(original)

    def refused_replace(src, dst):
        raise OSError(1, "simulated refusal to replace")

    with mock.patch.object(os, "replace", refused_replace):
        result = _edit("f.txt", "before", "after")
    assert result["isError"] is True and result["errorCategory"] == "validation"
    assert result["isRetryable"] is False
    assert target.read_bytes() == original
    assert _leftover_temporaries(ws) == []


def test_replace_failure_after_the_swap_is_reported_as_unknown_outcome(ws, monkeypatch):
    target = ws / "f.txt"
    target.write_bytes(b"before\n")
    real_replace = os.replace

    def replace_then_fail(src, dst):
        real_replace(src, dst)  # the swap happens, then the call reports an error
        raise OSError(1, "simulated error after the swap")

    with mock.patch.object(os, "replace", replace_then_fail):
        result = _edit("f.txt", "before", "after")
    assert result["isError"] is True and result["errorCategory"] == "unknown_outcome"
    assert result["isRetryable"] is False
    assert target.read_bytes() == b"after\n"


def test_permission_bits_are_preserved(ws):
    for mode in (0o640, 0o755, 0o600):
        target = ws / f"mode_{mode:o}.sh"
        target.write_bytes(b"echo one\n")
        os.chmod(target, mode)
        assert _edit(target.name, "one", "two")["isError"] is False
        assert stat.S_IMODE(os.stat(target).st_mode) == mode
        assert target.read_bytes() == b"echo two\n"


def test_crlf_line_endings_and_byte_order_mark_are_kept_as_stored(ws):
    target = ws / "win.txt"
    target.write_bytes(b"\xef\xbb\xbfone\r\ntwo\r\nthree")
    assert _edit("win.txt", "two\r\n", "TWO\r\n")["isError"] is False
    assert target.read_bytes() == b"\xef\xbb\xbfone\r\nTWO\r\nthree"


def test_lf_old_string_does_not_match_crlf_content(ws):
    target = ws / "win.txt"
    original = b"one\r\ntwo\r\n"
    target.write_bytes(original)
    result = _edit("win.txt", "one\ntwo\n", "x")
    assert result["isError"] is True and "CRLF" in result["content"]
    assert target.read_bytes() == original


def test_unicode_text_and_missing_trailing_newline_are_preserved(ws):
    target = ws / "u.txt"
    target.write_bytes("héllo wörld\nlast line without newline".encode("utf-8"))
    assert _edit("u.txt", "wörld", "🌍 world")["isError"] is False
    assert target.read_bytes() == "héllo 🌍 world\nlast line without newline".encode("utf-8")


def test_path_traversal_and_absolute_paths_are_refused(ws, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "secret.txt"
    outside.write_bytes(b"secret\n")
    for path in ("../" + outside.name, str(outside)):
        result = _edit(path, "secret", "changed")
        assert result["isError"] is True and result["errorCategory"] == "permission"
    assert outside.read_bytes() == b"secret\n"


def test_symlink_to_an_outside_file_is_refused(ws, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "secret.txt"
    outside.write_bytes(b"secret\n")
    (ws / "link.txt").symlink_to(outside)
    result = _edit("link.txt", "secret", "changed")
    assert result["isError"] is True and result["errorCategory"] == "permission"
    assert outside.read_bytes() == b"secret\n"
    assert (ws / "link.txt").is_symlink()


def test_symlink_to_an_inside_file_is_refused_too(ws):
    real = ws / "real.txt"
    real.write_bytes(b"content\n")
    (ws / "alias.txt").symlink_to(real)
    result = _edit("alias.txt", "content", "changed")
    assert result["isError"] is True and "symbolic link" in result["content"]
    assert real.read_bytes() == b"content\n"


def test_directory_symlink_escape_is_refused(ws, tmp_path_factory):
    outside_dir = tmp_path_factory.mktemp("outside_dir")
    (outside_dir / "x.txt").write_bytes(b"keep\n")
    (ws / "escape").symlink_to(outside_dir, target_is_directory=True)
    result = _edit("escape/x.txt", "keep", "changed")
    assert result["isError"] is True and result["errorCategory"] == "permission"
    assert (outside_dir / "x.txt").read_bytes() == b"keep\n"


def test_missing_file_is_not_created(ws):
    result = _edit("absent.txt", "a", "b")
    assert result["isError"] is True and "write_file" in result["content"]
    assert not (ws / "absent.txt").exists()


def test_directories_fifos_and_hard_links_are_refused(ws):
    (ws / "sub").mkdir()
    assert _edit("sub", "a", "b")["isError"] is True

    os.mkfifo(ws / "pipe")  # refused on type before anything reads it, so this cannot block
    assert _edit("pipe", "a", "b")["isError"] is True

    first = ws / "first.txt"
    first.write_bytes(b"shared\n")
    os.link(first, ws / "second.txt")
    result = _edit("first.txt", "shared", "changed")
    assert result["isError"] is True and "hard links" in result["content"]
    assert first.read_bytes() == b"shared\n" and (ws / "second.txt").read_bytes() == b"shared\n"


def test_binary_and_non_utf8_files_are_refused(ws):
    binary = ws / "blob.bin"
    binary.write_bytes(b"abc\x00def")
    latin = ws / "latin.txt"
    latin.write_bytes(b"caf\xe9\n")
    for name, old in (("blob.bin", "abc"), ("latin.txt", "caf")):
        result = _edit(name, old, "x")
        assert result["isError"] is True and result["errorCategory"] == "validation"
    assert binary.read_bytes() == b"abc\x00def" and latin.read_bytes() == b"caf\xe9\n"


def test_files_over_the_size_limit_are_refused(ws):
    target = ws / "big.txt"
    original = b"a" * (files.EDIT_MAX_FILE_BYTES + 1)
    target.write_bytes(original)
    result = _edit("big.txt", "a", "b")
    assert result["isError"] is True and "larger than" in result["content"]
    assert target.read_bytes() == original


def test_edit_file_is_a_file_tool_and_needs_no_grant(ws):
    (ws / "f.txt").write_bytes(b"one\n")
    result = asyncio.run(dispatch.dispatch_tool_async(
        "edit_file", {"path": "f.txt", "old_string": "one", "new_string": "two"},
        authorization=SessionAuthorization()))
    assert result["isError"] is False
    assert (ws / "f.txt").read_bytes() == b"two\n"
    assert local_registry().kind_of("edit_file") == FILE


def test_skill_allowlist_denies_edit_before_the_file_is_touched(ws):
    (ws / "f.txt").write_bytes(b"one\n")
    result = asyncio.run(dispatch.dispatch_tool_async(
        "edit_file", {"path": "f.txt", "old_string": "one", "new_string": "two"},
        allowed_tools=["read_file"], authorization=SessionAuthorization(allow_all=True)))
    assert result["isError"] is True and "not permitted" in result["content"]
    assert (ws / "f.txt").read_bytes() == b"one\n"


def test_dispatch_rejects_missing_arguments_without_touching_anything(ws):
    (ws / "f.txt").write_bytes(b"one\n")
    for tool_input in ({}, {"path": "f.txt"}, {"path": "f.txt", "old_string": "one"}):
        result = dispatch.dispatch_tool("edit_file", tool_input)
        assert result["isError"] is True
    assert (ws / "f.txt").read_bytes() == b"one\n"


def test_schema_declares_all_three_arguments_as_required():
    schema = next(t for t in get_local_tools() if t["function"]["name"] == "edit_file")["function"]
    assert schema["parameters"]["required"] == ["path", "old_string", "new_string"]
    assert set(schema["parameters"]["properties"]) == {"path", "old_string", "new_string"}


def _snapshot(root: Path):
    """Every entry under root with its kind and bytes: a refusal must leave all of it identical."""
    found = {}
    for path in sorted(root.rglob("*")):
        rel = str(path.relative_to(root))
        if path.is_symlink():
            found[rel] = ("link", os.readlink(path))
        elif path.is_file():
            found[rel] = ("file", path.read_bytes())
        else:
            found[rel] = ("dir", None)
    return found


@pytest.mark.parametrize("bad", ["bad\ud800", "\udc80tail", "mid\udfffdle"])
def test_a_lone_surrogate_in_replacement_text_is_a_controlled_refusal(ws, bad):
    target = ws / "f.txt"
    target.write_bytes(b"one\n")
    before = _snapshot(ws)
    result = dispatch.dispatch_tool("edit_file", {"path": "f.txt", "old_string": "one", "new_string": bad})
    assert result["isError"] is True and result["errorCategory"] == "validation"
    assert "not valid UTF-8" in result["content"] and result["isRetryable"] is True
    assert _snapshot(ws) == before
    assert _leftover_temporaries(ws) == []


def test_a_lone_surrogate_is_refused_even_when_old_string_does_not_match(ws):
    (ws / "f.txt").write_bytes(b"one\n")
    result = _edit("f.txt", "absent", "bad\ud800")
    assert result["isError"] is True and "not valid UTF-8" in result["content"]
    assert (ws / "f.txt").read_bytes() == b"one\n"


def test_a_direct_parent_directory_symlink_is_refused(ws):
    (ws / "real").mkdir()
    (ws / "real" / "x.txt").write_bytes(b"keep\n")
    (ws / "link").symlink_to(ws / "real", target_is_directory=True)
    before = _snapshot(ws)
    result = _edit("link/x.txt", "keep", "changed")
    assert result["isError"] is True and result["errorCategory"] == "permission"
    assert "passes through a symbolic link (link)" in result["content"]
    assert _snapshot(ws) == before


def test_a_nested_symlink_component_inside_the_workspace_is_refused(ws):
    (ws / "a" / "b").mkdir(parents=True)
    (ws / "a" / "b" / "x.txt").write_bytes(b"keep\n")
    (ws / "a" / "alias").symlink_to(ws / "a" / "b", target_is_directory=True)
    before = _snapshot(ws)
    result = _edit("a/alias/x.txt", "keep", "changed")
    assert result["isError"] is True and result["errorCategory"] == "permission"
    assert _snapshot(ws) == before


def test_an_escaping_parent_symlink_below_a_real_directory_is_refused(ws, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside")
    (outside / "x.txt").write_bytes(b"keep\n")
    (ws / "a").mkdir()
    (ws / "a" / "escape").symlink_to(outside, target_is_directory=True)
    before_ws, before_out = _snapshot(ws), _snapshot(outside)
    result = _edit("a/escape/x.txt", "keep", "changed")
    assert result["isError"] is True and result["errorCategory"] == "permission"
    assert _snapshot(ws) == before_ws and _snapshot(outside) == before_out


def test_dot_dot_after_a_symlink_is_refused_before_it_is_resolved(ws):
    (ws / "real").mkdir()
    (ws / "real" / "inner").mkdir()
    (ws / "f.txt").write_bytes(b"one\n")
    (ws / "real" / "inner" / "link").symlink_to(ws / "real", target_is_directory=True)
    before = _snapshot(ws)
    result = _edit("real/inner/link/../../f.txt", "one", "two")
    assert result["isError"] is True and result["errorCategory"] == "permission"
    assert _snapshot(ws) == before


def test_a_workspace_reached_through_a_symlinked_ancestor_is_legitimate(tmp_path, monkeypatch):
    real_outer = tmp_path / "real_outer"
    (real_outer / "ws" / "sub").mkdir(parents=True)
    (real_outer / "ws" / "sub" / "f.txt").write_bytes(b"one\n")
    (tmp_path / "outer").symlink_to(real_outer, target_is_directory=True)
    monkeypatch.chdir(tmp_path / "outer" / "ws")  # the root is entered through a link; it is the trust boundary
    result = _edit("sub/f.txt", "one", "two")
    assert result["isError"] is False
    assert (real_outer / "ws" / "sub" / "f.txt").read_bytes() == b"two\n"
