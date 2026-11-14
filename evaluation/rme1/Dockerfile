# RME-1.1 trial image. Contains the Agent Loop runtime, the agent-side runner, and the egress initializer.
# The hidden evaluator (evaluator.py), the host orchestration (harness.py, supervisor.py), and tests are NOT copied.
# Trial workspaces are bind-mounted at run time; nothing else from the host is mounted.
FROM python:3.13-slim
# The source revision is recorded in the image so the supervisor can verify what it runs
ARG REVISION=unknown
LABEL org.agent-loop.revision=$REVISION

RUN apt-get update && apt-get install -y --no-install-recommends procps iptables util-linux \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin agent

COPY requirements/base.txt /opt/req/base.txt
RUN pip install --no-cache-dir -r /opt/req/base.txt

COPY src /opt/agent-loop/src
COPY evaluation/rme1/scenarios.py evaluation/rme1/runner.py evaluation/rme1/probe.py evaluation/rme1/metrics.py evaluation/rme1/policy.py evaluation/rme1/egress-init.sh /opt/agent-loop/evaluation/rme1/
COPY evaluation/rme_debug/spec.py /opt/agent-loop/evaluation/rme_debug/spec.py
COPY evaluation/rme_debug_02/spec.py /opt/agent-loop/evaluation/rme_debug_02/spec.py
COPY evaluation/rme_tradeoff_01/spec.py /opt/agent-loop/evaluation/rme_tradeoff_01/spec.py

# Root-owned and not writable by the agent user
RUN chmod -R a-w /opt/agent-loop && find /opt/agent-loop -type d -exec chmod 755 {} + \
    && find /opt/agent-loop -type f -exec chmod 644 {} + && chmod 755 /opt/agent-loop/evaluation/rme1/egress-init.sh

WORKDIR /work
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/opt/agent-loop
# Root only for firewall initialization; egress-init.sh drops to uid 10001 before any other process starts
USER root
ENTRYPOINT ["/opt/agent-loop/evaluation/rme1/egress-init.sh"]
CMD ["/usr/local/bin/python", "/opt/agent-loop/evaluation/rme1/runner.py"]
