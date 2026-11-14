"""Processed-event record: the event ids whose application is recorded, in order. Ids are added with mark() and are
never removed by this module."""


class ProcessedSet:
    def __init__(self):
        self._ids = []
        self._members = set()

    def contains(self, event_id):
        return event_id in self._members

    def mark(self, event_id):
        if event_id not in self._members:
            self._members.add(event_id)
            self._ids.append(event_id)

    def ids(self):
        return list(self._ids)
