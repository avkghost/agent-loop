"""Idempotent event processor. The contract is in README.md."""

from enum import Enum

from .events import parse_event


class Outcome(str, Enum):
    APPLIED = "applied"
    DUPLICATE = "duplicate"


class AccountFrozen(Exception):
    """The account refuses changes while it is frozen. The event that hits it is not processed."""


class Processor:
    def __init__(self):
        self._balances = {}
        self._processed = set()
        self._ledger = []
        self._frozen = set()

    def freeze(self, account):
        self._frozen.add(account)

    def unfreeze(self, account):
        self._frozen.discard(account)

    def frozen_accounts(self):
        return sorted(self._frozen)

    def balance(self, account):
        return self._balances.get(account, 0)

    def ledger(self):
        return list(self._ledger)

    def apply(self, raw):
        event = parse_event(raw)
        if event.event_id in self._processed:
            return Outcome.DUPLICATE
        if event.account in self._frozen:
            raise AccountFrozen(f"account {event.account} is frozen")
        self._balances[event.account] = self.balance(event.account) + event.amount
        self._processed.add(event.event_id)
        self._ledger.append(event.event_id)
        return Outcome.APPLIED
