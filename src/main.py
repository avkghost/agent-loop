"""Main entry point for the agent loop."""

from . import cli


def main():
    """Delegate to CLI module for argument parsing and command routing."""
    cli.main()


if __name__ == "__main__":
    main()
