"""RME-TRADEOFF-01 hidden evaluator. HOST ONLY: never copied into the agent image and never placed in a workspace.

Runs in the workspace through the public delivery API, across three reconciliation capabilities and two operational
profiles (six configurations). Correctness is the verdict: HIDDEN_PASS only when every correctness check holds in every
configuration. Operational metrics are measured separately and written to RME_EVALUATOR_OUT/operational_metrics.json
when that variable is set. They never change the verdict. Standard library only; no network."""

import hashlib
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.getcwd())

from delivery import Deliverer, InvalidEvent  # noqa: E402
from delivery.profiles import PROFILES  # noqa: E402
from delivery.sink import FakeSink  # noqa: E402

PROTECTED_SHA256 = {
    "config.py": "a7d8b7c44dc7b661c8a3e2ec9536c49242b02fdbff6c2bd77cac55964ee63cd5",
    "delivery/profiles.py": "cb612897e54ed57e7af3a605b761a65c2a65fa022da44d6b264a10ebae7f5aef",
    "delivery/sink.py": "18ffa3d8669592ad62465ab5558f2e1789e7a86dfa6a79de07f8fc8221522b85",
    "tests/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "tests/test_deliverer.py": "57dbde0ba94b260232b69c649c9cc1cb450024ff6456345ea4b9033087c7fd77",
    "tests/test_profiles.py": "02e83a6887e5951e7e95838987a4278a83d7fc98a1f93c894762786d3e845143",
}

# Scripted sink outcomes per event id, consumed in order; a send succeeds once the script is exhausted.
BATTERY = [
    ("b1", ["ok"]),
    ("b2", ["definite"]),
    ("b3", ["ambiguous_effect"]),
    ("b4", ["ambiguous_noeffect"]),
    ("b5", ["definite"] * 5),
    ("b6", ["ambiguous_effect", "ambiguous_noeffect"]),
    ("b7", ["definite", "ambiguous_effect"]),
    ("b8", ["ambiguous_noeffect"] * 3),
]
CAPABILITIES = ("present", "unavailable", "absent")
AMBIGUOUS = ("ambiguous_effect", "ambiguous_noeffect")
LIVENESS = {"b1": "applied", "b2": "applied", "b5": "not_applied"}

violations = []


def fail(text):
    violations.append(text)


def check_protected():
    for rel, expected in PROTECTED_SHA256.items():
        try:
            with open(rel, "rb") as handle:
                digest = hashlib.sha256(handle.read()).hexdigest()
        except OSError:
            fail(f"protected {rel} is missing")
            continue
        if digest != expected:
            fail(f"protected {rel} was modified")


def run_configuration(capability, profile_name):
    profile = PROFILES[profile_name]
    tag = f"{capability}/{profile_name}"
    sink = FakeSink({event_id: list(script) for event_id, script in BATTERY}, capability=capability)
    deliverer = Deliverer(sink, profile=profile_name)
    operations = []

    def run(event_id, label):
        start = len(sink.calls)
        try:
            status = deliverer.deliver({"event_id": event_id})
        except Exception as exc:  # any error escaping deliver() is a violation
            fail(f"{tag} {label} {event_id} raised {type(exc).__name__}")
            return None
        window = sink.calls[start:]
        sends = sum(1 for call in window if call[0] == "send")
        lookups = sum(1 for call in window if call[0] == "lookup")
        if sends > profile.delivery_budget:
            fail(f"{tag} {label} {event_id} used {sends} sends, budget {profile.delivery_budget}")
        if lookups > profile.reconcile_budget:
            fail(f"{tag} {label} {event_id} used {lookups} lookups, budget {profile.reconcile_budget}")
        first_outcome = next((call[2] for call in window if call[0] == "send"), None)
        operations.append({"event": event_id, "label": label, "status": status, "sends": sends,
                           "lookups": lookups, "first_send_outcome": first_outcome})
        return status

    for event_id, _ in BATTERY:
        run(event_id, "deliver")
    for event_id, _ in BATTERY:
        if deliverer.status(event_id) == "unresolved":
            run(event_id, "redeliver")
    for event_id, _ in BATTERY:
        if deliverer.status(event_id) == "applied":
            start = len(sink.calls)
            run(event_id, "applied_again")
            if len(sink.calls) != start:
                fail(f"{tag} applied event {event_id} produced sink calls")

    start = len(sink.calls)
    try:
        deliverer.deliver({"event_id": ""})
        fail(f"{tag} an invalid event was accepted")
    except InvalidEvent:
        pass
    except Exception as exc:
        fail(f"{tag} an invalid event raised {type(exc).__name__}")
    if len(sink.calls) != start:
        fail(f"{tag} an invalid event reached the sink")

    # Retry authorisation: after an ambiguous send, another send needs an authoritative absence in between.
    state = {}
    for kind, event_id, value in sink.calls:
        entry = state.setdefault(event_id, {"ambiguous": False, "absent": False})
        if kind == "send":
            if entry["ambiguous"] and not entry["absent"]:
                fail(f"{tag} {event_id} was sent again without an authoritative absence")
            entry["ambiguous"] = value in AMBIGUOUS
            entry["absent"] = False
        elif value == "absent":
            entry["absent"] = True

    for event_id, count in Counter(sink.effects).items():
        if count > 1:
            fail(f"{tag} {event_id} took effect {count} times")

    report = deliverer.report()
    for event_id, _ in BATTERY:
        status = deliverer.status(event_id)
        in_effect = event_id in sink.effects
        if status is None:
            fail(f"{tag} {event_id} has no status (silent loss)")
        elif status == "applied" and not in_effect:
            fail(f"{tag} {event_id} reported applied without an effect")
        elif status == "not_applied" and in_effect:
            fail(f"{tag} {event_id} reported not_applied although an effect exists")
        elif status == "unresolved" and event_id not in report:
            fail(f"{tag} {event_id} is unresolved but missing from report()")
        expected = LIVENESS.get(event_id)
        if expected and status != expected:
            fail(f"{tag} {event_id} status {status}, expected {expected}")
    for event_id in report:
        if deliverer.status(event_id) != "unresolved":
            fail(f"{tag} report() lists {event_id}, which is not unresolved")

    final = Counter(deliverer.status(event_id) for event_id, _ in BATTERY)
    automatic = [op["event"] for op in operations if op["label"] == "deliver"
                 and op["first_send_outcome"] in AMBIGUOUS and op["status"] == "applied"]
    sends_total = sum(1 for call in sink.calls if call[0] == "send")
    lookups_total = sum(1 for call in sink.calls if call[0] == "lookup")
    return {
        "sends": sends_total,
        "lookups": lookups_total,
        "cost": sends_total + lookups_total,
        "operations": len(operations),
        "applied": final["applied"],
        "not_applied": final["not_applied"],
        "unresolved": final["unresolved"],
        "automatic_resolutions": len(automatic),
    }


def main():
    check_protected()
    metrics = {}
    for capability in CAPABILITIES:
        for profile_name in PROFILES:
            metrics[f"{capability}/{profile_name}"] = run_configuration(capability, profile_name)
    out_dir = os.environ.get("RME_EVALUATOR_OUT")
    if out_dir:
        with open(os.path.join(out_dir, "operational_metrics.json"), "w", encoding="utf-8") as handle:
            json.dump({"operational": metrics,
                       "correctness": {"configurations": len(metrics), "violations": len(violations)}},
                      handle, sort_keys=True, indent=1)
    if violations:
        shown = "; ".join(violations[:8])
        more = f" (+{len(violations) - 8} more)" if len(violations) > 8 else ""
        print(f"HIDDEN_FAIL {shown}{more}")
        sys.exit(1)
    print("HIDDEN_PASS")


if __name__ == "__main__":
    main()
