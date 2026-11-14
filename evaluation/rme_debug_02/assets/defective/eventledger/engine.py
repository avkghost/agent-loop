"""Engine: applies one event at a time as a unit of work. The contract is in README.md."""

from enum import Enum

from .dedupe import ProcessedSet
from .intake import parse_event
from .ledger import Ledger
from .state import AccountState
from .store import FakeStore, StorageError


class Outcome(str, Enum):
    APPLIED = "applied"
    DUPLICATE = "duplicate"


class Engine:
    def __init__(self, store=None):
        self._store = store if store is not None else FakeStore()
        self._state = AccountState()
        self._ledger = Ledger()
        self._processed = ProcessedSet()

    @property
    def store(self):
        return self._store

    def freeze(self, account):
        self._state.freeze(account)

    def unfreeze(self, account):
        self._state.unfreeze(account)

    def balance(self, account):
        return self._state.balance(account)

    def balances(self):
        return self._state.balances()

    def ledger(self):
        return self._ledger.entries()

    def processed_ids(self):
        return self._processed.ids()

    def apply(self, raw):
        event = parse_event(raw)
        if self._processed.contains(event.event_id):
            return Outcome.DUPLICATE
        self._state.require_open(event.account)
        state_before = self._state.snapshot()
        ledger_before = self._ledger.length()
        self._state.credit(event.account, event.amount)
        self._ledger.append(event)
        self._processed.mark(event.event_id)
        try:
            self._store.commit(self._image())
        except StorageError:
            self._state.restore(state_before)
            self._ledger.truncate(ledger_before)
            raise
        return Outcome.APPLIED

    def _image(self, extra=None):
        processed = self._processed.ids() + ([extra] if extra is not None else [])
        return {"balances": self._state.balances(), "ledger": self._ledger.entries(), "processed": processed}
