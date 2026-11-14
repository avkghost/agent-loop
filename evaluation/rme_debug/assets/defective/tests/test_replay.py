import unittest

from ledger import Processor, replay


class ReplayTests(unittest.TestCase):
    def test_summary_counts_outcomes_and_reports_invalid_lines(self):
        lines = [
            '{"event_id": "e1", "account": "a", "amount": 2}',
            '{"event_id": "e1", "account": "a", "amount": 2}',
            "not json",
            '{"event_id": "", "account": "a", "amount": 2}',
        ]
        processor = Processor()
        summary = replay(lines, processor)
        self.assertEqual((summary["applied"], summary["duplicate"], summary["invalid"], summary["failed"]), (1, 1, 2, 0))
        self.assertEqual(processor.balance("a"), 2)
        self.assertEqual(len(summary["errors"]), 2)

    def test_blank_lines_are_skipped_and_errors_name_their_line(self):
        summary = replay(["", '{"event_id": "b1", "account": "a", "amount": 1}', "  ", "[1]"], Processor())
        self.assertEqual((summary["applied"], summary["invalid"]), (1, 1))
        self.assertTrue(summary["errors"][0].startswith("line 4:"))
