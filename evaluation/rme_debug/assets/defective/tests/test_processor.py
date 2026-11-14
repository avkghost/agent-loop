import unittest

from ledger import AccountFrozen, InvalidEvent, Outcome, Processor


def event(event_id, account="acc-1", amount=5):
    return {"event_id": event_id, "account": account, "amount": amount}


class ProcessorTests(unittest.TestCase):
    def test_first_delivery_is_applied(self):
        processor = Processor()
        self.assertEqual(processor.apply(event("e1")), Outcome.APPLIED)
        self.assertEqual(processor.balance("acc-1"), 5)

    def test_duplicate_delivery_is_not_applied_twice(self):
        processor = Processor()
        processor.apply(event("e1"))
        self.assertEqual(processor.apply(event("e1")), Outcome.DUPLICATE)
        self.assertEqual(processor.balance("acc-1"), 5)
        self.assertEqual(processor.ledger(), ["e1"])

    def test_distinct_events_are_processed_independently(self):
        processor = Processor()
        self.assertEqual(processor.apply(event("e1", amount=5)), Outcome.APPLIED)
        self.assertEqual(processor.apply(event("e2", amount=5)), Outcome.APPLIED)
        self.assertEqual(processor.balance("acc-1"), 10)

    def test_failed_operation_is_not_recorded_and_can_be_retried(self):
        processor = Processor()
        processor.freeze("acc-1")
        with self.assertRaises(AccountFrozen):
            processor.apply(event("e1"))
        self.assertEqual(processor.balance("acc-1"), 0)
        self.assertEqual(processor.ledger(), [])
        processor.unfreeze("acc-1")
        self.assertEqual(processor.apply(event("e1")), Outcome.APPLIED)
        self.assertEqual(processor.balance("acc-1"), 5)

    def test_freezing_one_account_does_not_affect_another(self):
        processor = Processor()
        processor.freeze("acc-1")
        self.assertEqual(processor.apply(event("e9", account="acc-2")), Outcome.APPLIED)
        self.assertEqual(processor.frozen_accounts(), ["acc-1"])

    def test_ledger_lists_applied_events_in_order(self):
        processor = Processor()
        for event_id in ["e1", "e2", "e1", "e3"]:
            processor.apply(event(event_id))
        self.assertEqual(processor.ledger(), ["e1", "e2", "e3"])
        self.assertEqual(processor.balance("acc-1"), 15)

    def test_invalid_event_is_rejected_and_not_recorded(self):
        processor = Processor()
        with self.assertRaises(InvalidEvent):
            processor.apply({"event_id": "e1", "account": "acc-1", "amount": True})
        self.assertEqual(processor.apply(event("e1")), Outcome.APPLIED)

    def test_negative_amounts_reduce_the_balance(self):
        processor = Processor()
        processor.apply(event("p1", amount=10))
        self.assertEqual(processor.apply(event("p2", amount=-4)), Outcome.APPLIED)
        self.assertEqual(processor.balance("acc-1"), 6)
