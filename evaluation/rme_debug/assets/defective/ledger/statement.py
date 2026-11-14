"""Balance statement: a deterministic view of account balances and freeze state."""


def statement(processor, accounts):
    """Rows sorted by account name: account, balance and frozen. An account with no events has a zero balance."""
    frozen = set(processor.frozen_accounts())
    return [{"account": name, "balance": processor.balance(name), "frozen": name in frozen}
            for name in sorted(set(accounts))]


def format_statement(rows):
    """Fixed-width text: a header, one line per account, then a total line. Equal input gives equal output."""
    width = max([len(row["account"]) for row in rows] + [len("account")])
    header = "account".ljust(width) + "  " + "balance".rjust(10) + "  frozen"
    lines = [header]
    total = 0
    for row in rows:
        total += row["balance"]
        flag = "yes" if row["frozen"] else "no"
        lines.append(row["account"].ljust(width) + "  " + str(row["balance"]).rjust(10) + "  " + flag)
    lines.append("total".ljust(width) + "  " + str(total).rjust(10))
    return "\n".join(lines) + "\n"
