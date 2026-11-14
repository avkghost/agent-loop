"""Simulated persistence. commit() stores a deep copy of the image it is given, or raises StorageError. A commit whose
number (counted from 1) is listed in fail_commits fails and stores nothing. This stands in for a real database; it
provides no durability across a crash and no concurrency control."""

import copy


class StorageError(Exception):
    """The commit did not reach storage. Nothing from that commit is stored."""


class FakeStore:
    def __init__(self, fail_commits=()):
        self._calls = 0
        self._fail = set(fail_commits)
        self.committed = []

    @property
    def calls(self):
        return self._calls

    def commit(self, image):
        self._calls += 1
        if self._calls in self._fail:
            raise StorageError(f"commit {self._calls} failed")
        self.committed.append(copy.deepcopy(image))
