import unittest

from eventledger import Engine, reconcile


class ReportTests(unittest.TestCase):
    def test_a_normal_sequence_reconciles_cleanly(self):
        engine = Engine()
        engine.apply({"event_id": "r1", "account": "a", "amount": 4})
        engine.apply({"event_id": "r2", "account": "b", "amount": 6})
        engine.apply({"event_id": "r1", "account": "a", "amount": 4})
        self.assertEqual(reconcile(engine), [])
