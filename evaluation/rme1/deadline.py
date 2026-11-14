"""RME host-side deadline handling. HOST ONLY (never copied into the image).

run_with_deadline() runs one command under a wall-clock budget:
- On the deadline it sends SIGTERM to the client process. For `docker run` with a TTY-less attached client, Docker
  proxies the signal to the container's init, so the runner can finish gracefully and publish its frames.
- After `grace` seconds it forces termination: the optional `force_remove` callable runs (for containers,
  `docker rm -f`), and the whole process group is killed. Nothing is left running.
- Output goes to files, not pipes, so a large or blocked stream cannot stall the deadline.

SnapshotPoller copies the in-container session out on an interval and keeps only the last snapshot that passes the
existing session validator. A failed copy or an invalid document never replaces the previous valid snapshot.
"""

import json
import os
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

GRACE_SECONDS = 20.0


class SnapshotSourceGone(Exception):
    """The container no longer exists (removed after exit). Not a rejected copy: the last valid snapshot is kept."""


@dataclass
class DeadlineResult:
    stdout: str
    stderr: str
    returncode: Optional[int]
    timed_out: bool
    graceful: Optional[bool]   # True: exited after SIGTERM; False: forced; None: did not time out
    elapsed: float
    kill_error: Optional[str] = None  # set when the forced group kill was refused or the client survived it


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def run_with_deadline(argv: List[str], deadline: float, grace: float = GRACE_SECONDS,
                      force_remove: Optional[Callable[[], None]] = None, cwd: Optional[str] = None,
                      env: Optional[dict] = None) -> DeadlineResult:
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="rme-deadline-") as tmp:
        out_path, err_path = Path(tmp) / "stdout", Path(tmp) / "stderr"
        with open(out_path, "wb") as out, open(err_path, "wb") as err:
            proc = subprocess.Popen(argv, stdout=out, stderr=err, cwd=cwd, env=env, start_new_session=True)
            try:
                rc = proc.wait(timeout=deadline)
                return DeadlineResult(_read(out_path), _read(err_path), rc, False, None,
                                      round(time.monotonic() - started, 1))
            except subprocess.TimeoutExpired:
                pass
            # Deadline reached: ask nicely first
            try:
                os.kill(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            graceful = True
            kill_error = None
            try:
                proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                graceful = False
                if force_remove is not None:
                    force_remove()
                kill_error = _force_group(proc)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    kill_error = (kill_error or "") + "; client still running after SIGKILL"
        return DeadlineResult(_read(out_path), _read(err_path), proc.returncode, True, graceful,
                              round(time.monotonic() - started, 1), kill_error)


def _force_group(proc: subprocess.Popen) -> Optional[str]:
    """SIGKILL the child's process group. A refused signal (EPERM) is reported, never raised: the caller records
    it, and the container removal already happened. The direct child is killed as a fallback."""
    error = None
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError as exc:
        error = f"group kill refused: {exc}"
        if proc.poll() is None:
            proc.kill()
    return error


class SnapshotPoller:
    """Keeps the last valid session snapshot on the host. `copy_out(dest_path) -> bool` copies the in-container
    session to dest_path and returns False if no file exists yet. `validate(dict)` raises on an invalid document."""

    def __init__(self, dest: Path, copy_out: Callable[[Path], bool], validate: Callable[[dict], object],
                 interval: float = 5.0):
        self.dest = dest
        self._copy_out = copy_out
        self._validate = validate
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self.status = {"copies": 0, "valid": 0, "rejected": 0, "last_error": None,
                       "last_valid_at": None, "last_saved_at": None, "last_lag_seconds": None,
                       "last_rejection": None, "source_gone": 0}

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=self._interval + 10)
        self.poll_once()  # one last attempt after the process has ended

    def _loop(self):
        while not self._stop.wait(self._interval):
            self.poll_once()

    def poll_once(self) -> bool:
        self.status["copies"] += 1
        fd, tmp_name = tempfile.mkstemp(prefix="snapshot-", suffix=".json", dir=str(self.dest.parent))
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            if not self._copy_out(tmp):
                self.status["last_error"] = "no session file yet"
                return False
            document = json.loads(tmp.read_text(encoding="utf-8"))
            self._validate(document)
            os.replace(tmp, self.dest)  # atomic: the previous valid snapshot is replaced only by a valid one
            now = time.time()
            saved = document.get("saved_at")
            self.status["valid"] += 1
            self.status["last_error"] = None
            self.status["last_valid_at"] = round(now, 3)
            self.status["last_saved_at"] = saved
            # Lag: how old the saved state was when it was copied out (the session's own timestamp vs. the copy)
            if isinstance(saved, str):
                # The session writes a naive datetime.now() inside the image, whose clock is UTC (no TZ is set).
                # Interpret it as UTC explicitly; the host's local zone is not the writer's zone.
                from datetime import datetime, timezone
                written = datetime.fromisoformat(saved).replace(tzinfo=timezone.utc).timestamp()
                self.status["last_lag_seconds"] = round(now - written, 2)
            return True
        except SnapshotSourceGone as exc:
            self.status["source_gone"] += 1
            self.status["last_error"] = f"source gone: {str(exc)[:160]}"
            return False
        except Exception as exc:  # an invalid copy is rejected and the previous snapshot is kept
            self.status["rejected"] += 1
            self.status["last_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
            self.status["last_rejection"] = self.status["last_error"]  # kept even after a later 'no file' poll
            return False
        finally:
            if tmp.exists():
                tmp.unlink()
