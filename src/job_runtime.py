"""Worker thread with its own event loop, and the lifecycle gate that governs it.

WorkerRuntime owns one thread and one event loop, with a bounded executor for blocking calls.
RuntimeGate is the only place runtime state changes. Its lock guards transitions only and is never held
while waiting on the worker, so the worker can never deadlock against a caller.
"""

import asyncio
import concurrent.futures
import threading
from typing import Callable, Optional, Tuple

RUNNING = "running"
CLOSING = "closing"
CLOSED = "closed"

BLOCKING_WORKERS = 4  # bound on blocking calls (such as ps inspections) run from the worker


class RuntimeRefused(Exception):
    """The runtime is not accepting work. The message is safe to show to the model."""


class WorkerRuntime:
    def __init__(self, name: str):
        self.loop = asyncio.new_event_loop()
        self.loop.set_default_executor(
            concurrent.futures.ThreadPoolExecutor(max_workers=BLOCKING_WORKERS, thread_name_prefix=f"{name}-io"))
        self.thread = threading.Thread(target=self._serve, name=name, daemon=True)
        self.thread.start()

    def _serve(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
        leftovers = asyncio.all_tasks(self.loop)
        for task in leftovers:
            task.cancel()
        if leftovers:
            self.loop.run_until_complete(asyncio.gather(*leftovers, return_exceptions=True))
        self.loop.run_until_complete(self.loop.shutdown_default_executor())
        self.loop.close()

    def submit(self, coro) -> Optional[concurrent.futures.Future]:
        """Schedule a coroutine on the worker. None if the worker has already stopped (the coroutine is closed)."""
        try:
            return asyncio.run_coroutine_threadsafe(coro, self.loop)
        except RuntimeError:
            coro.close()
            return None

    def stop(self, timeout: float) -> bool:
        """Stop the loop and join the thread. True when the thread is gone."""
        if not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout)
        return not self.thread.is_alive()


class RuntimeGate:
    """Linearizable runtime lifecycle: never started, running, closing, closed.

    - acquire() creates the runtime on first use and refuses while closing or closed.
    - begin_shutdown() moves running to closing. Exactly one shutdown attempt runs at a time; callers that
      find an attempt in progress are told to wait. A closing runtime whose earlier attempt did not complete
      can be retried, but never by a second worker: the same runtime object is always reused.
    - end_shutdown(completed) moves closing to closed only when the attempt completed, which means the
      worker thread has actually stopped. An incomplete attempt leaves the gate closing, still refusing work.
    - reopen() is the only way back to a fresh runtime. It is refused while closing, while an attempt is in
      progress, and while the current worker thread is alive.

    The lock guards state transitions only. It is never held while waiting for the worker or for an attempt.
    """

    def __init__(self, factory: Callable[[], object]):
        self._factory = factory
        self._lock = threading.Lock()
        self._runtime = None
        self._state: Optional[str] = None  # None: never started
        self._performing = False
        self._attempt_ended = threading.Event()
        self._attempt_ended.set()

    def acquire(self):
        with self._lock:
            if self._state is None:
                self._runtime = self._factory()
                self._state = RUNNING
            if self._state == RUNNING:
                return self._runtime
            if self._state == CLOSING:
                raise RuntimeRefused("Managed jobs are unavailable: the runtime is shutting down. No job was started.")
            raise RuntimeRefused("Managed jobs are unavailable: the runtime has shut down.")

    def running(self):
        with self._lock:
            return self._runtime if self._state == RUNNING else None

    def current(self):
        """The runtime object regardless of state (for inspection), or None if none was ever started."""
        with self._lock:
            return self._runtime

    def state(self) -> Optional[str]:
        with self._lock:
            return self._state

    def begin_shutdown(self) -> Tuple[object, bool]:
        """(runtime, performs). performs is True for exactly one caller at a time."""
        with self._lock:
            if self._state is None:
                return None, False
            if self._state == RUNNING:
                self._state = CLOSING
            elif self._state == CLOSED:
                return self._runtime, False
            if self._performing:
                return self._runtime, False
            self._performing = True
            self._attempt_ended.clear()
            return self._runtime, True

    def end_shutdown(self, completed: bool) -> None:
        with self._lock:
            self._performing = False
            if completed:
                self._state = CLOSED
            self._attempt_ended.set()

    def wait_attempt(self, timeout: float) -> bool:
        """Wait for the current attempt to end. Holds no lock while waiting. True if it ended in time."""
        return self._attempt_ended.wait(timeout)

    def reopen(self) -> None:
        with self._lock:
            if self._state == CLOSING or self._performing:
                raise RuntimeError("cannot reopen the runtime while its shutdown is incomplete")
            if self._runtime is not None and self._runtime.worker_alive():
                raise RuntimeError("cannot reopen the runtime while its worker thread is alive")
            self._state = None
            self._runtime = None
