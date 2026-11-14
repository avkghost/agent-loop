FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

COPY requirements/base.txt requirements/dev.txt requirements.txt ./requirements/
RUN apt-get update \
    && apt-get install --no-install-recommends --yes git procps \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir -r requirements/dev.txt

COPY . .

RUN useradd --create-home --uid 10001 agent \
    && chown -R agent:agent /workspace
USER agent

ENTRYPOINT ["python", "-m", "pytest"]
CMD ["-p", "no:cacheprovider", "-q", "-rfEXs", "-W", "default", "tests"]
