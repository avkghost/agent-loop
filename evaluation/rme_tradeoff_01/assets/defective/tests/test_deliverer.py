import unittest

from delivery import APPLIED, Deliverer, FakeSink, InvalidEvent


def ev(event_id):
    return {"event_id": event_id}


class DelivererTests(unittest.TestCase):
    def test_a_successful_delivery_is_applied_once(self):
        sink = FakeSink()
        self.assertEqual(Deliverer(sink, profile="live").deliver(ev("e1")), APPLIED)
        self.assertEqual(sink.effects, ["e1"])

    def test_a_definite_failure_is_retried_within_the_budget(self):
        sink = FakeSink({"e2": ["definite"]})
        self.assertEqual(Deliverer(sink, profile="live").deliver(ev("e2")), APPLIED)
        self.assertEqual(sink.effects, ["e2"])

    def test_an_ambiguous_timeout_with_effect_does_not_create_a_second_effect(self):
        sink = FakeSink({"e3": ["ambiguous_effect"]})
        Deliverer(sink, profile="live").deliver(ev("e3"))
        self.assertEqual(sink.effects, ["e3"])

    def test_an_ambiguous_timeout_without_effect_creates_at_most_one_effect(self):
        sink = FakeSink({"e4": ["ambiguous_noeffect"]})
        Deliverer(sink, profile="safe").deliver(ev("e4"))
        self.assertLessEqual(len(sink.effects), 1)

    def test_every_delivered_event_has_a_status_and_unresolved_events_are_reported(self):
        sink = FakeSink({"e5": ["ambiguous_effect"], "e6": ["ambiguous_noeffect"]}, capability="absent")
        deliverer = Deliverer(sink, profile="live")
        for event_id in ["e5", "e6"]:
            deliverer.deliver(ev(event_id))
        for event_id in ["e5", "e6"]:
            status = deliverer.status(event_id)
            self.assertIn(status, ("applied", "not_applied", "unresolved"))
            self.assertLessEqual(sink.effects.count(event_id), 1)
            if status == "unresolved":
                self.assertIn(event_id, deliverer.report())

    def test_an_invalid_event_is_rejected_without_contacting_the_sink(self):
        sink = FakeSink()
        with self.assertRaises(InvalidEvent):
            Deliverer(sink, profile="live").deliver({"event_id": ""})
        self.assertEqual(sink.calls, [])

    def test_a_delivered_event_is_not_sent_again(self):
        sink = FakeSink()
        deliverer = Deliverer(sink, profile="live")
        deliverer.deliver(ev("e7"))
        deliverer.deliver(ev("e7"))
        self.assertEqual(sink.effects, ["e7"])
