"""Batch application. Applies events in input order, one unit of work each, and reports the outcome of every event.
A failed or rejected event is reported and does not stop the batch. It stays eligible for a later batch."""

from dataclasses import dataclass, field

from .engine import Outcome
from .intake import InvalidEvent
from .state import AccountFrozen
from .store import StorageError


@dataclass
class BatchReport:
    results: list = field(default_factory=list)

    def count(self, status):
        return sum(1 for _, found, _ in self.results if found == status)


def apply_batch(engine, raw_events):
    report = BatchReport()
    for index, raw in enumerate(raw_events):
        try:
            outcome = engine.apply(raw)
        except InvalidEvent as exc:
            report.results.append((index, "invalid", str(exc)))
        except AccountFrozen as exc:
            report.results.append((index, "frozen", str(exc)))
        except StorageError as exc:
            report.results.append((index, "failed", str(exc)))
        else:
            report.results.append((index, outcome.value, ""))
    return report
