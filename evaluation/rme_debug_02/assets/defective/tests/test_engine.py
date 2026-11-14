import unittest

from eventledger import AccountFrozen, Engine, FakeStore, InvalidEvent, Outcome, StorageError


def event(event_id, account="acc-1", amount=5):
    return {"event_id": event_id, "account": account, "amount": amount}


class EngineTests(unittest.TestCase):
    def test_an_event_is_applied_once_and_redelivery_is_a_duplicate(self):
        engine = Engine()
        self.assertEqual(engine.apply(event("e1")), Outcome.APPLIED)
        self.assertEqual(engine.apply(event("e1")), Outcome.DUPLICATE)
        self.assertEqual(engine.balance("acc-1"), 5)

    def test_distinct_events_are_processed_independently(self):
        engine = Engine()
        self.assertEqual(engine.apply(event("e1")), Outcome.APPLIED)
        self.assertEqual(engine.apply(event("e2")), Outcome.APPLIED)
        self.assertEqual(engine.balance("acc-1"), 10)

    def test_a_successful_commit_records_the_ledger_and_the_processed_ids(self):
        engine = Engine()
        engine.apply(event("e1"))
        engine.apply(event("e2", account="acc-2", amount=7))
        self.assertEqual([entry["event_id"] for entry in engine.ledger()], ["e1", "e2"])
        self.assertEqual(engine.processed_ids(), ["e1", "e2"])
        self.assertEqual(engine.store.committed[-1]["processed"], ["e1", "e2"])

    def test_a_frozen_account_rejects_the_event_without_changing_anything(self):
        engine = Engine()
        engine.freeze("acc-1")
        with self.assertRaises(AccountFrozen):
            engine.apply(event("e1"))
        self.assertEqual((engine.balance("acc-1"), engine.ledger(), engine.processed_ids()), (0, [], []))
        self.assertEqual(engine.store.committed, [])

    def test_a_storage_failure_propagates_and_leaves_the_balance_unchanged(self):
        engine = Engine(store=FakeStore(fail_commits={1}))
        with self.assertRaises(StorageError):
            engine.apply(event("e1"))
        self.assertEqual(engine.balance("acc-1"), 0)

    def test_a_failed_event_is_applied_once_when_redelivered(self):
        engine = Engine(store=FakeStore(fail_commits={1}))
        with self.assertRaises(StorageError):
            engine.apply(event("e1"))
        self.assertEqual(engine.apply(event("e1")), Outcome.APPLIED)
        self.assertEqual(engine.balance("acc-1"), 5)

    def test_an_invalid_event_is_rejected_and_not_applied(self):
        engine = Engine()
        with self.assertRaises(InvalidEvent):
            engine.apply({"event_id": "e1", "account": "acc-1", "amount": True})
        self.assertEqual(engine.apply(event("e1")), Outcome.APPLIED)
