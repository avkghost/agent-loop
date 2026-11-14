"""RME-DEBUG-02 hidden evaluator. HOST ONLY: never copied into the agent image and never placed in a workspace.
Runs with the workspace as the working directory and uses only the public eventledger API. Prints HIDDEN_PASS only when
every check passes. Standard library only; no network."""

import hashlib
import os
import sys

sys.path.insert(0, os.getcwd())

from eventledger import AccountFrozen, Engine, FakeStore, InvalidEvent, Outcome, StorageError, apply_batch, reconcile  # noqa: E402

PROTECTED_SHA256 = {
    "tests/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "tests/test_batch.py": "bbe98ed4faf2c383a1ec6c41c8e901c52dea7296b1d1b5200c9fd444b3d0d527",
    "tests/test_engine.py": "dcfecbb7544e3ba06cc9fae9a42b71af4d3a2661971e55e0210afe39bf584cc6",
    "tests/test_intake.py": "aaa51e30ec6804e6563dba2ec7525e758d5aa29416f0a1783d9ed347f66f32cd",
    "tests/test_report.py": "3b773a3721165364a3c29a3dd31eeb9c49ff1d5140fbf8b429839746595079ff",
    "tests/test_store.py": "db275d3ef9f536fa5a8cb48f9b4bd1a29f3dfbf078ad12c3bb90176e4f3fb0af",
}

failures = []


def check(name, condition):
    if not condition:
        failures.append(name)


def raises(exc_type, fn):
    try:
        fn()
    except exc_type:
        return True
    return False


def ev(event_id, account="A", amount=5):
    return {"event_id": event_id, "account": account, "amount": amount}


for rel, expected in PROTECTED_SHA256.items():
    with open(rel, "rb") as handle:
        check(f"protected {rel} unchanged", hashlib.sha256(handle.read()).hexdigest() == expected)

# F1. Frozen account: rejected before any change; no commit; redelivery after unfreezing applies once.
f = Engine()
f.freeze("F")
check("F1 frozen raises AccountFrozen", raises(AccountFrozen, lambda: f.apply(ev("f1", "F", 3))))
check("F1 nothing changed", f.balance("F") == 0 and f.ledger() == [] and f.processed_ids() == [])
check("F1 no commit attempted", f.store.calls == 0)
f.unfreeze("F")
check("F1 redelivery applies once", f.apply(ev("f1", "F", 3)) == Outcome.APPLIED and f.balance("F") == 3)

# F2. Storage failure at commit: propagates; everything rolled back; nothing persisted; redelivery applies once.
s = Engine(store=FakeStore(fail_commits={1}))
persisted_before = list(s.store.committed)
check("F2 StorageError propagates", raises(StorageError, lambda: s.apply(ev("g1", "G", 4))))
check("F2 balance rolled back", s.balance("G") == 0)
check("F2 ledger rolled back", s.ledger() == [])
check("F2 not recorded as processed", s.processed_ids() == [])
check("F2 nothing persisted", s.store.committed == persisted_before)
check("F2 consistent after the failure", reconcile(s) == [])
check("F2 redelivery applies", s.apply(ev("g1", "G", 4)) == Outcome.APPLIED)
check("F2 balance after redelivery", s.balance("G") == 4)
check("F2 second delivery is a duplicate", s.apply(ev("g1", "G", 4)) == Outcome.DUPLICATE)
check("F2 balance applied exactly once", s.balance("G") == 4)
check("F2 duplicate made no commit", s.store.calls == 2)

# B. Batch application reports a failed commit and keeps going; a later batch redelivers the failed event.
b = Engine(store=FakeStore(fail_commits={1}))
first = apply_batch(b, [ev("k1", "K", 2), ev("k2", "K", 3)])
check("B first batch reports failed then applied", [status for _, status, _ in first.results] == ["failed", "applied"])
second = apply_batch(b, [ev("k1", "K", 2)])
check("B later batch applies the failed event", [status for _, status, _ in second.results] == ["applied"])
check("B balance after both batches", b.balance("K") == 5 and reconcile(b) == [])

# F3. Repeated storage failures, then success.
r = Engine(store=FakeStore(fail_commits={1, 2}))
check("F3 first failure", raises(StorageError, lambda: r.apply(ev("h1", "H", 2))))
check("F3 second failure", raises(StorageError, lambda: r.apply(ev("h1", "H", 2))))
check("F3 third attempt applies", r.apply(ev("h1", "H", 2)) == Outcome.APPLIED and r.balance("H") == 2)
check("F3 consistent", reconcile(r) == [])

# D. Distinct events are independent, including around a failure.
d = Engine(store=FakeStore(fail_commits={1}))
raises(StorageError, lambda: d.apply(ev("x1", "D", 5)))
check("D distinct event with the same amount applies", d.apply(ev("x2", "D", 5)) == Outcome.APPLIED)
check("D balance counts only committed events", d.balance("D") == 5)
check("D failed event applies on redelivery", d.apply(ev("x1", "D", 5)) == Outcome.APPLIED)
check("D balance after both", d.balance("D") == 10)
check("D same id, different amount, is a duplicate",
      d.apply(ev("x2", "D", 99)) == Outcome.DUPLICATE and d.balance("D") == 10)
check("D ledger order", [entry["event_id"] for entry in d.ledger()] == ["x2", "x1"])
check("D consistent", reconcile(d) == [])

# Q. Idempotency after success on an account that has had a failure.
q = Engine(store=FakeStore(fail_commits={1}))
raises(StorageError, lambda: q.apply(ev("y1", "Q", 7)))
check("Q retry applies", q.apply(ev("y1", "Q", 7)) == Outcome.APPLIED)
check("Q redelivery after success is a duplicate", q.apply(ev("y1", "Q", 7)) == Outcome.DUPLICATE)
check("Q balance applied once", q.balance("Q") == 7)
check("Q later distinct event on the same account applies",
      q.apply(ev("y2", "Q", 1)) == Outcome.APPLIED and q.balance("Q") == 8)

# V. Validation and error propagation preserved.
v = Engine()
check("V invalid amount raises InvalidEvent",
      raises(InvalidEvent, lambda: v.apply({"event_id": "v1", "account": "A", "amount": 1.5})))
check("V invalid event never reaches the store", v.store.calls == 0 and v.ledger() == [] and v.processed_ids() == [])
check("V valid event after an invalid one applies", v.apply(ev("v1")) == Outcome.APPLIED)

# C. Consistency through an ordinary sequence.
c = Engine()
for item in [ev("c1", "P", 3), ev("c2", "R", 4), ev("c1", "P", 3), ev("c3", "P", 2)]:
    c.apply(item)
check("C ledger has each id once, in order", [entry["event_id"] for entry in c.ledger()] == ["c1", "c2", "c3"])
check("C balances match applied amounts", c.balance("P") == 5 and c.balance("R") == 4)
check("C consistent", reconcile(c) == [])
check("C last committed image equals live state", c.store.committed[-1]["processed"] == c.processed_ids())

if failures:
    print("HIDDEN_FAIL " + "; ".join(failures))
    sys.exit(1)
print("HIDDEN_PASS")
