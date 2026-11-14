"""Append-only ledger of applied events. Entries are copies, so a caller cannot change the record. truncate() is used by
the engine to remove the entry of a unit of work that did not commit."""


class Ledger:
    def __init__(self):
        self._entries = []

    def append(self, event):
        self._entries.append({"event_id": event.event_id, "account": event.account, "amount": event.amount})

    def entries(self):
        return [dict(entry) for entry in self._entries]

    def length(self):
        return len(self._entries)

    def truncate(self, length):
        del self._entries[length:]
