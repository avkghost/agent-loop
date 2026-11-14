"""RME-DEBUG-01 hidden evaluator. HOST ONLY: never copied into the agent image and never placed in a workspace.
It runs with the workspace as the working directory and uses only the public ledger API. Prints HIDDEN_PASS only when
every check passes. Standard library only; no network."""

import hashlib
import os
import sys

sys.path.insert(0, os.getcwd())

from ledger import AccountFrozen, InvalidEvent, Outcome, Processor, replay  # noqa: E402

PROTECTED_SHA256 = {
    "tests/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "tests/test_cli.py": "f145396da925715780636834d58cb424e16322eefef91c567c5889bcc25b66bf",
    "tests/test_events.py": "66f11233bf5c4cc50ea04e225337bb82a56b2132dc473e725deca5c281599cfc",
    "tests/test_processor.py": "e602b8a4e9018ac5f85b4002f944c1aaf871705d160a18a92ca3b3b5a742a834",
    "tests/test_replay.py": "15b84cd427e840d095e1f4f202600eeebebe87804e0abf446076ffa315bf15c5",
    "tests/test_statement.py": "bed94729ff6bcfb966f5f5c8ceafbed141d9728bba18f5c79a1aad3e3b583357",
}

failures = []


def check(name, condition):
    if not condition:
        failures.append(name)


def event(event_id, account="A", amount=5):
    return {"event_id": event_id, "account": account, "amount": amount}


def raises(exc_type, fn):
    try:
        fn()
    except exc_type:
        return True
    return False


# Protected benchmark assets are unchanged.
for rel, expected in PROTECTED_SHA256.items():
    with open(rel, "rb") as handle:
        check(f"protected {rel} unchanged", hashlib.sha256(handle.read()).hexdigest() == expected)

# H1. Duplicate delivery never applies twice, including after an earlier failure on the same account.
p = Processor()
check("H1 first delivery applied", p.apply(event("x1")) == Outcome.APPLIED)
check("H1 duplicate reported", p.apply(event("x1")) == Outcome.DUPLICATE)
check("H1 balance applied once", p.balance("A") == 5)
p.freeze("B")
check("H1 frozen delivery fails", raises(AccountFrozen, lambda: p.apply(event("y1", "B", 7))))
p.unfreeze("B")
check("H1 retry after failure applied", p.apply(event("y1", "B", 7)) == Outcome.APPLIED)
check("H1 redelivery after success is duplicate", p.apply(event("y1", "B", 7)) == Outcome.DUPLICATE)
check("H1 balance of B applied once", p.balance("B") == 7)

# H2. Distinct events are independent, even with the same account and amount.
check("H2 distinct event with same amount applied", p.apply(event("x2")) == Outcome.APPLIED)
check("H2 balance reflects both distinct events", p.balance("A") == 10)
check("H2 same id with different amount is a duplicate",
      p.apply(event("x2", amount=99)) == Outcome.DUPLICATE and p.balance("A") == 10)

# H3. Failures propagate, change nothing, and are not recorded.
q = Processor()
q.freeze("C")
before = (q.balance("C"), q.ledger())
check("H3 failure propagates", raises(AccountFrozen, lambda: q.apply(event("z1", "C"))))
check("H3 failure leaves balance and ledger unchanged", (q.balance("C"), q.ledger()) == before)
q.unfreeze("C")
check("H3 event applied once after the account is unfrozen", q.apply(event("z1", "C")) == Outcome.APPLIED)
check("H3 balance after the retry", q.balance("C") == 5)

# H4. State consistency: each applied id appears once, in order; balances equal the applied sums.
r = Processor()
for item in [event("a1", "P", 3), event("a2", "Q", 4), event("a1", "P", 3), event("a3", "P", 2)]:
    r.apply(item)
check("H4 ledger has each applied id once, in order", r.ledger() == ["a1", "a2", "a3"])
check("H4 balances equal applied sums", r.balance("P") == 5 and r.balance("Q") == 4)

# H5. Preserved behaviour: invalid events are rejected and unrecorded; a freeze affects only its account;
# replay reports the outcomes of an ordinary sequence.
s = Processor()
check("H5 invalid amount raises InvalidEvent",
      raises(InvalidEvent, lambda: s.apply({"event_id": "v1", "account": "A", "amount": 1.5})))
check("H5 invalid event not recorded", s.apply(event("v1")) == Outcome.APPLIED)
t = Processor()
t.freeze("X")
check("H5 freeze does not block other accounts", t.apply(event("o1", "Y")) == Outcome.APPLIED)
summary = replay(['{"event_id": "r1", "account": "A", "amount": 1}',
                  '{"event_id": "r1", "account": "A", "amount": 1}'], Processor())
check("H5 replay counts applied and duplicate", (summary["applied"], summary["duplicate"]) == (1, 1))

if failures:
    print("HIDDEN_FAIL " + "; ".join(failures))
    sys.exit(1)
print("HIDDEN_PASS")
