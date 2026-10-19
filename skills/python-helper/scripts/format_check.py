#!/usr/bin/env python3
"""Check Python file formatting and style."""

import sys
import os

def check_format(filepath):
    """Basic Python format checker."""
    if not filepath.endswith('.py'):
        print(f"Error: Not a Python file: {filepath}")
        return 1

    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return 1

    with open(filepath) as f:
        content = f.read()

    # Basic checks
    lines = content.split('\n')
    issues = []

    for i, line in enumerate(lines, 1):
        if line and len(line) > 100:
            issues.append(f"  Line {i}: Line too long ({len(line)} chars)")
        if line.endswith(' '):
            issues.append(f"  Line {i}: Trailing whitespace")

    if issues:
        print(f"\n⚠️  Style issues found in {filepath}:")
        for issue in issues[:10]:  # Show first 10
            print(issue)
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
        return 1
    else:
        print(f"✅ {filepath}: No obvious style issues")
        return 0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 format_check.py <file>")
        sys.exit(1)
    sys.exit(check_format(sys.argv[1]))
