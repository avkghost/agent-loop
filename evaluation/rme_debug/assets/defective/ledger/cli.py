"""Command line: replay a JSON-lines events file, print the summary, and print a statement for the named accounts.

Usage: python3 -m ledger EVENTS_FILE [ACCOUNT ...]
"""

import json
import sys

from .processor import Processor
from .replay import replay
from .statement import format_statement, statement

USAGE = "usage: python3 -m ledger EVENTS_FILE [ACCOUNT ...]\n"


def main(argv=None, stdout=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    stdout = stdout or sys.stdout
    if not argv:
        stdout.write(USAGE)
        return 2
    path, accounts = argv[0], argv[1:]
    with open(path, encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    processor = Processor()
    summary = replay(lines, processor)
    counts = {key: summary[key] for key in ("applied", "duplicate", "failed", "invalid")}
    stdout.write(json.dumps(counts, sort_keys=True) + "\n")
    if accounts:
        stdout.write(format_statement(statement(processor, accounts)))
    return 0
