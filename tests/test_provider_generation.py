"""Generation parameter ownership: LM Studio's sampling settings apply unless Agent Loop is explicitly configured.

Offline only. Requests are built and inspected; nothing is sent to a model.
"""

import math

import pytest

from src import config, provider, responses_provider
from src.config_loader import apply_config_overrides

ALLOWED_CHAT_KEYS = {"model", "messages", "temperature", "max_tokens", "stream", "tools", "tool_choice"}
FORBIDDEN_KEYS = {"top_p", "top_k", "min_p", "repeat_penalty", "repetition_penalty", "frequency_penalty",
                  "presence_penalty", "seed", "stop", "reasoning", "reasoning_budget", "reasoning_effort"}
MESSAGES = [{"role": "user", "content": "hello"}]


@pytest.fixture(autouse=True)
def _restore_config(monkeypatch):
    monkeypatch.setattr(config, "TEMPERATURE", config.TEMPERATURE)
    monkeypatch.setattr(config, "REQUEST_MAX_TOKENS", config.REQUEST_MAX_TOKENS)


# --- temperature ---------------------------------------------------------------------------------------------------

def test_default_request_omits_temperature_so_lm_studio_applies_its_own_setting():
    payload = provider.build_payload(MESSAGES, None, provider.GenerationOptions())
    assert "temperature" not in payload
    assert set(payload) == {"model", "messages", "max_tokens", "stream"}


def test_default_request_with_tools_adds_only_the_tool_fields():
    payload = provider.build_payload(MESSAGES, [{"type": "function", "function": {"name": "t"}}],
                                     provider.GenerationOptions())
    assert "temperature" not in payload
    assert set(payload) == {"model", "messages", "max_tokens", "stream", "tools", "tool_choice"}


def test_an_explicit_temperature_is_transmitted_exactly():
    payload = provider.build_payload(MESSAGES, None, provider.GenerationOptions(temperature=0.4))
    assert payload["temperature"] == 0.4


@pytest.mark.parametrize("value", [0, 0.0, 2, 2.0, 0.7])
def test_temperature_boundaries_and_ordinary_values_are_accepted(value):
    assert provider.GenerationOptions(temperature=value).temperature == float(value)


@pytest.mark.parametrize("value", [True, False, "0.2", [0.2], -0.1, 2.01, 3, math.nan, math.inf, -math.inf])
def test_invalid_temperatures_are_rejected_before_a_request_is_built(value):
    with pytest.raises(ValueError, match="temperature"):
        provider.GenerationOptions(temperature=value)


def test_validate_temperature_accepts_none_as_omitted():
    assert provider.validate_temperature(None) is None


# --- output limit --------------------------------------------------------------------------------------------------

def test_the_default_output_limit_is_8192():
    assert config.REQUEST_MAX_TOKENS == 8192
    assert provider.GenerationOptions().max_tokens == 8192
    assert provider.build_payload(MESSAGES, None, provider.GenerationOptions())["max_tokens"] == 8192


def test_a_configured_output_limit_is_applied_to_later_requests(monkeypatch):
    monkeypatch.setattr(config, "REQUEST_MAX_TOKENS", 1024)
    assert provider.build_payload(MESSAGES, None, provider.GenerationOptions())["max_tokens"] == 1024


@pytest.mark.parametrize("value", [0, -1, 1.5, "8192", None, True])
def test_invalid_output_limits_are_rejected(value):
    with pytest.raises(ValueError, match="REQUEST_MAX_TOKENS"):
        provider.GenerationOptions(max_tokens=value)


def test_the_conversation_budget_is_not_the_output_limit():
    assert config.MAX_TOKENS != config.REQUEST_MAX_TOKENS
    assert provider.GenerationOptions().max_tokens == config.REQUEST_MAX_TOKENS


# --- configuration file -------------------------------------------------------------------------------------------

def test_yaml_sets_the_output_limit_and_it_is_validated():
    apply_config_overrides({"REQUEST_MAX_TOKENS": "2048"})
    assert config.REQUEST_MAX_TOKENS == 2048
    for bad in ("0", "-5", "abc", "1.5"):
        with pytest.raises(ValueError):
            apply_config_overrides({"REQUEST_MAX_TOKENS": bad})
    assert config.REQUEST_MAX_TOKENS == 2048, "a refused value must not replace the configured one"


def test_yaml_sets_temperature_and_null_restores_the_default():
    apply_config_overrides({"TEMPERATURE": "0.4"})
    assert provider.GenerationOptions().temperature == 0.4
    apply_config_overrides({"TEMPERATURE": "null"})
    assert config.TEMPERATURE is None
    assert "temperature" not in provider.build_payload(MESSAGES, None, provider.GenerationOptions())


@pytest.mark.parametrize("value", ["3", "-1", "warm", "true"])
def test_invalid_yaml_temperatures_are_rejected(value):
    with pytest.raises((ValueError, TypeError)):
        apply_config_overrides({"TEMPERATURE": value})
    assert config.TEMPERATURE is None


# --- compaction keeps its own explicit options ---------------------------------------------------------------------

def test_compaction_options_are_unchanged_by_the_default_temperature():
    options = provider.GenerationOptions(temperature=0.3, max_tokens=500, stream=False)
    payload = provider.build_payload(MESSAGES, None, options)
    assert payload == {"model": config.MODEL_NAME, "messages": MESSAGES, "temperature": 0.3,
                       "max_tokens": 500, "stream": False}


# --- no unintended fields, in either adapter -------------------------------------------------------------------------

def test_no_sampling_reasoning_or_penalty_field_is_introduced_by_default_or_explicit_settings():
    for options in (provider.GenerationOptions(), provider.GenerationOptions(temperature=0.4)):
        payload = provider.build_payload(MESSAGES, [{"type": "function", "function": {"name": "t"}}], options)
        assert set(payload) <= ALLOWED_CHAT_KEYS
        assert not FORBIDDEN_KEYS & set(payload)


def test_responses_request_omits_temperature_by_default_and_sends_it_when_set():
    default = responses_provider.build_request(MESSAGES, None, provider.GenerationOptions())
    assert "temperature" not in default
    assert default["max_output_tokens"] == 8192
    explicit = responses_provider.build_request(MESSAGES, None, provider.GenerationOptions(temperature=0.4))
    assert explicit["temperature"] == 0.4
    for body in (default, explicit):
        assert not FORBIDDEN_KEYS & set(body)


# --- the primary loop sends what the configuration says -------------------------------------------------------------

def test_the_loop_request_omits_temperature_by_default(run_loop):
    from tests.conftest import make_streamed_response
    post, _ = run_loop([make_streamed_response(finish_reason="stop", content="done")])
    assert "temperature" not in post.payloads[0]
    assert post.payloads[0]["max_tokens"] == 8192


def test_the_loop_request_carries_a_configured_temperature(run_loop):
    from tests.conftest import make_streamed_response
    post, _ = run_loop([make_streamed_response(finish_reason="stop", content="done")], TEMPERATURE=0.4)
    assert post.payloads[0]["temperature"] == 0.4
