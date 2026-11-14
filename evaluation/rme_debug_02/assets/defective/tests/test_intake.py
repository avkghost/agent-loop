import unittest

from eventledger import Event, InvalidEvent, parse_event


class IntakeTests(unittest.TestCase):
    def test_a_valid_event_parses_into_its_fields(self):
        self.assertEqual(parse_event({"event_id": "e", "account": "a", "amount": -2}), Event("e", "a", -2))

    def test_each_violation_of_the_contract_is_rejected(self):
        invalid = [
            "not an object",
            {"account": "a", "amount": 1},
            {"event_id": "", "account": "a", "amount": 1},
            {"event_id": "e", "account": "", "amount": 1},
            {"event_id": "e", "account": "a"},
            {"event_id": "e", "account": "a", "amount": "1"},
            {"event_id": "e", "account": "a", "amount": True},
        ]
        for raw in invalid:
            with self.subTest(raw=raw):
                with self.assertRaises(InvalidEvent):
                    parse_event(raw)
