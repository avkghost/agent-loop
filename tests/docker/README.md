# Linux subprocess test environment

This harness builds the current source tree into a non-root Python 3.13 Linux container and runs pytest without host mounts, network access, extra capabilities, or a shared PID namespace. Docker's `--init` is enabled so PID 1 reaps container children.

From the repository root:

```sh
tests/docker/run-linux-tests.sh
tests/docker/run-linux-tests.sh tests/test_shell_hardening.py
```

The default container limits are 128 processes, 1 GiB RAM, and 2 CPUs. The image installs `requirements/dev.txt`; no LLM, MCP server, network service, Docker socket, SSH agent, or host home directory is available.

The tests verify Linux subprocess behavior, process groups, stdin isolation, bounded output, timeout/cancellation cleanup, and Phase A journal integration. They do not prove behavior under host kernel failures, PID reuse races, power loss, or detached descendants that create a new session.
