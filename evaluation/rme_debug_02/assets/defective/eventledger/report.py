"""Consistency check over an engine and its store. Returns a list of problems; an empty list means consistent."""


def reconcile(engine):
    problems = []
    ledger = engine.ledger()
    totals = {}
    for entry in ledger:
        totals[entry["account"]] = totals.get(entry["account"], 0) + entry["amount"]
    for account in sorted(set(totals) | set(engine.balances())):
        if totals.get(account, 0) != engine.balance(account):
            problems.append(f"balance of {account} does not equal the ledger total")
    ledger_ids = [entry["event_id"] for entry in ledger]
    if len(set(ledger_ids)) != len(ledger_ids):
        problems.append("a ledger entry appears more than once")
    if ledger_ids != engine.processed_ids():
        problems.append("processed records do not match the ledger")
    committed = getattr(engine.store, "committed", [])
    if committed:
        last = committed[-1]
        if last["ledger"] != ledger or last["processed"] != engine.processed_ids():
            problems.append("the last committed image differs from the live state")
    return problems
