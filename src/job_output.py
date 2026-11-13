"""Bounded job output and opaque, job-bound cursors.

OutputBuffer keeps at most `limit` bytes per stream and addresses them by absolute offsets, so a reader
can tell how much it missed. Cursors are authenticated: a cursor is valid only for the job that issued it,
and a forged, altered, or foreign cursor is rejected with the same answer.
"""

import base64
import binascii
import hashlib
import hmac
import secrets
from collections import deque
from typing import Optional, Tuple

READ_CHUNK = 64 << 10
_COALESCE = 64 << 10
_MAC_BYTES = 16
_MAX_CURSOR_CHARS = 256

# Per-process key: cursors do not outlive the runtime that issued them.
_CURSOR_KEY = secrets.token_bytes(32)


class OutputBuffer:
    """Bounded byte buffer addressed by absolute offsets. The oldest bytes are dropped first."""

    def __init__(self, limit: int):
        self.limit = limit
        self.chunks: deque = deque()
        self.start = 0   # absolute offset of the oldest retained byte
        self.end = 0     # absolute offset just past the newest byte
        self.size = 0

    def append(self, data: bytes) -> None:
        if not data:
            return
        if self.chunks and len(self.chunks[-1]) < _COALESCE:
            self.chunks[-1] = self.chunks[-1] + data
        else:
            self.chunks.append(bytes(data))
        self.end += len(data)
        self.size += len(data)
        while self.size > self.limit:
            excess = self.size - self.limit
            head = self.chunks[0]
            if len(head) <= excess:
                self.chunks.popleft()
                self.size -= len(head)
                self.start += len(head)
            else:
                self.chunks[0] = head[excess:]
                self.size -= excess
                self.start += excess

    def read(self, cursor: int, limit: int):
        """(data, next_cursor, dropped_bytes, more). A cursor older than the buffer reports what was dropped."""
        dropped = 0
        if cursor < self.start:
            dropped = self.start - cursor
            cursor = self.start
        stop = min(self.end, cursor + limit)
        data = self._slice(cursor, stop)
        next_cursor = cursor + len(data)
        return data, next_cursor, dropped, next_cursor < self.end

    def _slice(self, a: int, b: int) -> bytes:
        out = bytearray()
        pos = self.start
        for chunk in self.chunks:
            c0, c1 = pos, pos + len(chunk)
            pos = c1
            if c1 <= a:
                continue
            if c0 >= b:
                break
            out += chunk[max(a, c0) - c0:min(b, c1) - c0]
        return bytes(out)


def _mac(body: bytes) -> bytes:
    return hmac.new(_CURSOR_KEY, body, hashlib.sha256).digest()[:_MAC_BYTES]


def encode_cursor(job_id: str, stdout_offset: int, stderr_offset: int) -> str:
    """An opaque token bound to one job and the next byte to read on each stream."""
    body = f"{job_id}|{stdout_offset}|{stderr_offset}".encode()
    return base64.urlsafe_b64encode(body + _mac(body)).decode().rstrip("=")


def decode_cursor(token: object, job_id: str) -> Optional[Tuple[int, int]]:
    """(stdout_offset, stderr_offset) if the token is authentic and was issued for job_id. Otherwise None.

    Every failure returns None, so a caller cannot tell a forged token from a foreign one.
    """
    if not isinstance(token, str) or not token or len(token) > _MAX_CURSOR_CHARS:
        return None
    try:
        raw = base64.b64decode(token + "=" * (-len(token) % 4), altchars=b"-_", validate=True)
    except (binascii.Error, ValueError):
        return None
    if len(raw) <= _MAC_BYTES:
        return None
    body, mac = raw[:-_MAC_BYTES], raw[-_MAC_BYTES:]
    if not hmac.compare_digest(_mac(body), mac):
        return None
    try:
        issued_for, so, se = body.decode().split("|")
        stdout_offset, stderr_offset = int(so), int(se)
    except (UnicodeDecodeError, ValueError):
        return None
    if issued_for != job_id or stdout_offset < 0 or stderr_offset < 0:
        return None
    return stdout_offset, stderr_offset
