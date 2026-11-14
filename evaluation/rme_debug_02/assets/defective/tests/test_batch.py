import unittest

from eventledger import Engine, apply_batch


class BatchTests(unittest.TestCase):
    def test_a_batch_reports_each_event_in_order(self):
        engine = Engine()
        report = apply_batch(engine, [
            {"event_id": "b1", "account": "a", "amount": 1},
            {"event_id": "b1", "account": "a", "amount": 1},
            {"event_id": "b2", "account": "a", "amount": True},
        ])
        self.assertEqual([status for _, status, _ in report.results], ["applied", "duplicate", "invalid"])
        self.assertEqual(engine.balance("a"), 1)

    def test_a_frozen_event_is_reported_and_does_not_stop_the_batch(self):
        engine = Engine()
        engine.freeze("z")
        report = apply_batch(engine, [{"event_id": "c1", "account": "z", "amount": 2},
                                      {"event_id": "c2", "account": "y", "amount": 3}])
        self.assertEqual([status for _, status, _ in report.results], ["frozen", "applied"])
        self.assertEqual(report.count("applied"), 1)
