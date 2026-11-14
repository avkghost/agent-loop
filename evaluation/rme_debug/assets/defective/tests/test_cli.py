import io
import json
import os
import tempfile
import unittest

from ledger import Processor, cli, replay


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "events.jsonl")
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write('{"event_id": "c1", "account": "x", "amount": 2}\n')
            handle.write('{"event_id": "c1", "account": "x", "amount": 2}\n')
            handle.write('{"event_id": "c2", "account": "y", "amount": 5}\n')

    def tearDown(self):
        self.tmp.cleanup()

    def test_prints_the_summary_and_the_requested_statement(self):
        out = io.StringIO()
        self.assertEqual(cli.main([self.path, "x", "y"], stdout=out), 0)
        lines = out.getvalue().splitlines()
        self.assertEqual(json.loads(lines[0]), {"applied": 2, "duplicate": 1, "failed": 0, "invalid": 0})
        self.assertTrue(lines[1].startswith("account"))
        self.assertTrue(lines[-1].endswith("7"))

    def test_without_arguments_prints_usage_and_returns_two(self):
        out = io.StringIO()
        self.assertEqual(cli.main([], stdout=out), 2)
        self.assertIn("usage", out.getvalue())

    def test_failed_events_for_a_frozen_account_are_counted(self):
        processor = Processor()
        processor.freeze("x")
        counts = replay(['{"event_id": "f1", "account": "x", "amount": 1}'], processor)
        self.assertEqual((counts["failed"], counts["applied"]), (1, 0))
        self.assertEqual(processor.balance("x"), 0)
