"""Deterministic model provider for tests. No network: every model request is answered by a script.

Tests script the model by replacing POST, a module-level function with the shape of an HTTP call:
POST(url, json=payload, timeout=..., stream=bool) returns a response-like object (raise_for_status,
iter_lines for streams, json() for non-streamed answers, close). The provider builds the payload exactly as
the real adapter does, runs the response through the same SSE parser, and maps failures to typed turns.
Unscripted use fails loudly, so a test can never silently reach a real endpoint.
"""

import copy
import json

import requests

from src import config
from src import provider as p
from src import stream as sse


def _unscripted(url, json=None, timeout=None, stream=False):
    raise AssertionError("a test reached the model boundary without a scripted reply")


POST = _unscripted  # tests replace this to script the model


class ScriptedProvider:
    def __init__(self):
        self.payloads = []

    async def complete(self, messages, tools, options, on_text=None):
        payload = p.build_payload(messages, tools, options)
        self.payloads.append(copy.deepcopy(payload))
        try:
            response = POST(config.LM_STUDIO_URL, json=payload,
                            timeout=(config.LM_STUDIO_CONNECT_TIMEOUT, config.LM_STUDIO_TIMEOUT), stream=options.stream)
        except requests.exceptions.Timeout as e:
            return p.failure(p.TurnOutcome.TIMEOUT, "timeout", str(e))
        except Exception as e:  # connection refused, DNS failure, and similar transport errors
            return p.failure(p.TurnOutcome.TRANSPORT, "transport", f"{type(e).__name__}: {e}")
        try:
            return self._read(response, options, on_text)
        finally:
            close = getattr(response, "close", None)
            if close is not None:
                close()

    def _read(self, response, options, on_text):
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            return p.failure(p.TurnOutcome.HTTP_ERROR, "http", str(e))
        try:
            if not options.stream:
                return p.turn_from_json(response.json())
            parser = sse.StreamParser(on_text)
            for line in response.iter_lines():
                parser.feed(line)
            return p.turn_from_parser(parser)
        except sse.StreamError as e:
            return p.turn_from_stream_error(e)
        except requests.exceptions.Timeout as e:
            return p.failure(p.TurnOutcome.TIMEOUT, "timeout", str(e))
        except requests.exceptions.RequestException as e:
            return p.failure(p.TurnOutcome.TRANSPORT, "transport", f"{type(e).__name__}: {e}")
        except ValueError as e:
            return p.failure(p.TurnOutcome.MALFORMED, "malformed", f"response is not JSON: {e}")

    async def aclose(self):
        return None


def install(monkeypatch):
    """Make the runtime use a ScriptedProvider. Each test scripts the model by replacing POST."""
    monkeypatch.setattr(p, "make_provider", lambda: ScriptedProvider())
