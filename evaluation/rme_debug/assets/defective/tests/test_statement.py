import unittest

from ledger import Processor, format_statement, statement


class StatementTests(unittest.TestCase):
    def setUp(self):
        self.processor = Processor()
        self.processor.apply({"event_id": "s1", "account": "beta", "amount": 3})
        self.processor.apply({"event_id": "s2", "account": "alpha", "amount": 4})
        self.processor.freeze("beta")

    def test_rows_are_sorted_and_report_balance_and_freeze_state(self):
        rows = statement(self.processor, ["beta", "alpha", "gamma"])
        self.assertEqual([row["account"] for row in rows], ["alpha", "beta", "gamma"])
        self.assertEqual(rows[1], {"account": "beta", "balance": 3, "frozen": True})
        self.assertEqual(rows[2], {"account": "gamma", "balance": 0, "frozen": False})

    def test_format_is_deterministic_with_a_total_line(self):
        text = format_statement(statement(self.processor, ["beta", "alpha"]))
        self.assertEqual(text, format_statement(statement(self.processor, ["alpha", "beta"])))
        lines = text.splitlines()
        self.assertTrue(lines[0].startswith("account"))
        self.assertTrue(lines[-1].startswith("total"))
        self.assertTrue(lines[-1].endswith("7"))

    def test_empty_statement_has_a_header_and_a_zero_total(self):
        lines = format_statement(statement(Processor(), [])).splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("account"))
        self.assertTrue(lines[1].startswith("total") and lines[1].endswith("0"))

    def test_frozen_flag_follows_unfreeze(self):
        self.processor.unfreeze("beta")
        rows = statement(self.processor, ["beta"])
        self.assertEqual(rows, [{"account": "beta", "balance": 3, "frozen": False}])
