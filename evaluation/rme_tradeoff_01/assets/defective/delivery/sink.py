"""Downstream sink (protected test double). Deterministic and offline.

Each event id has an optional script of send outcomes, consumed in order. Once the script is exhausted, a send succeeds.
  ok                  the effect is applied and the call returns
  definite            the call fails and nothing is applied
  ambiguous_effect    the effect is applied, then the call times out: the caller cannot tell
  ambiguous_noeffect  the call times out and nothing is applied: the caller cannot tell
Reconciliation capability:
  present      lookup(event_id) returns True if the effect exists, False if not
  unavailable  lookup(event_id) raises ReconciliationUnavailable
  absent       lookup is None: the sink offers no reconciliation
"""


class DefiniteFailure(Exception):
    """The send failed and nothing was applied."""


class AmbiguousTimeout(Exception):
    """The send timed out. Whether the effect was applied is unknown."""


class ReconciliationUnavailable(Exception):
    """Reconciliation could not answer. This never authorises a send."""


OUTCOMES = ("ok", "definite", "ambiguous_effect", "ambiguous_noeffect")
CAPABILITIES = ("present", "unavailable", "absent")


class FakeSink:
    def __init__(self, scripts=None, capability="present"):
        if capability not in CAPABILITIES:
            raise ValueError(f"unknown capability {capability!r}")
        self._scripts = {event_id: list(script) for event_id, script in (scripts or {}).items()}
        self._capability = capability
        self.effects = []  # one entry per applied effect, in order
        self.calls = []    # ("send", event_id, outcome) and ("lookup", event_id, result), in call order
        self.lookup = None if capability == "absent" else self._lookup

    def send(self, event_id):
        script = self._scripts.get(event_id)
        outcome = script.pop(0) if script else "ok"
        if outcome not in OUTCOMES:
            raise ValueError(f"unknown scripted outcome {outcome!r}")
        self.calls.append(("send", event_id, outcome))
        if outcome in ("ok", "ambiguous_effect"):
            self.effects.append(event_id)
        if outcome == "definite":
            raise DefiniteFailure(f"send {event_id} failed")
        if outcome.startswith("ambiguous"):
            raise AmbiguousTimeout(f"send {event_id} timed out")

    def _lookup(self, event_id):
        if self._capability == "unavailable":
            self.calls.append(("lookup", event_id, "unavailable"))
            raise ReconciliationUnavailable("reconciliation is unavailable")
        present = event_id in self.effects
        self.calls.append(("lookup", event_id, "present" if present else "absent"))
        return present
