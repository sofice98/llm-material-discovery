# -*- coding: utf-8 -*-
"""Shared OpenAI-compatible Responses API client."""
from __future__ import annotations

import atexit
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESPONSES_URL = "https://api.minimaxi.com/v1/responses"
RETRYABLE_STATUS_CODES = {408, 409, 425, 429}
_TOKEN_USAGE_LOCK = threading.Lock()
_TOTAL_REGULAR_INPUT_TOKENS = 0
_TOTAL_CACHED_INPUT_TOKENS = 0
_TOTAL_OUTPUT_TOKENS = 0
_TOKEN_SUMMARY_REGISTERED = False


class LLMClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResponsesClient:
    api_key: str
    url: str
    model: str
    timeout_seconds: int
    max_retries: int

    def respond(
        self,
        input_data: str | list[dict[str, Any]],
        *,
        max_output_tokens: int,
        reasoning_effort: str = "none",
    ) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "input": input_data,
            "max_output_tokens": max_output_tokens,
        }
        if reasoning_effort:
            body["reasoning"] = {"effort": reasoning_effort}

        payload = self._post_with_retries(body)
        accumulate_token_usage(payload)
        return extract_response_text(payload)

    def _post_with_retries(self, body: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    self.url,
                    headers=headers,
                    json=body,
                    timeout=self.timeout_seconds,
                )
                if (
                    response.status_code in RETRYABLE_STATUS_CODES
                    or response.status_code >= 500
                ) and attempt < self.max_retries:
                    time.sleep(2**attempt)
                    continue
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise LLMClientError("Responses API response must be a JSON object.")
                return payload
            except (requests.RequestException, ValueError, LLMClientError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(2**attempt)
                    continue
                response_body = ""
                if isinstance(exc, requests.RequestException) and exc.response is not None:
                    response_body = f"; response body: {exc.response.text[:1000]}"
                raise LLMClientError(f"Responses API request failed: {exc}{response_body}") from exc
        raise LLMClientError(f"Responses API request failed: {last_error}")


def create_client() -> ResponsesClient:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("API_KEY", "").strip()
    model = os.getenv("MODEL", "").strip()
    url = resolve_responses_url()
    timeout_seconds = positive_int_env("MODEL_TIMEOUT_SECONDS", 600)
    max_retries = nonnegative_int_env("MODEL_MAX_RETRIES", 2)
    missing = [name for name, value in (("API_KEY", api_key), ("MODEL", model)) if not value]
    if missing:
        raise LLMClientError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". Please create .env from .env.example."
        )
    return ResponsesClient(api_key, url, model, timeout_seconds, max_retries)


def resolve_responses_url() -> str:
    configured = os.getenv("MODEL_URL", "").strip()
    if configured:
        return convert_to_responses_url(configured)

    return DEFAULT_RESPONSES_URL

def convert_to_responses_url(url: str) -> str:
    normalized = url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized[: -len("/chat/completions")] + "/responses"
    return normalized


def accumulate_token_usage(payload: dict[str, Any]) -> None:
    global _TOTAL_REGULAR_INPUT_TOKENS, _TOTAL_CACHED_INPUT_TOKENS
    global _TOTAL_OUTPUT_TOKENS, _TOKEN_SUMMARY_REGISTERED

    usage = payload.get("usage")
    input_tokens = token_count(usage.get("input_tokens")) if isinstance(usage, dict) else 0
    output_tokens = token_count(usage.get("output_tokens")) if isinstance(usage, dict) else 0
    input_details = usage.get("input_tokens_details") if isinstance(usage, dict) else None
    cached_tokens = (
        token_count(input_details.get("cached_tokens"))
        if isinstance(input_details, dict)
        else 0
    )
    cached_tokens = min(cached_tokens, input_tokens)
    regular_input_tokens = input_tokens - cached_tokens
    with _TOKEN_USAGE_LOCK:
        if not _TOKEN_SUMMARY_REGISTERED:
            atexit.register(print_total_token_usage)
            _TOKEN_SUMMARY_REGISTERED = True
        _TOTAL_REGULAR_INPUT_TOKENS += regular_input_tokens
        _TOTAL_CACHED_INPUT_TOKENS += cached_tokens
        _TOTAL_OUTPUT_TOKENS += output_tokens


def print_total_token_usage() -> None:
    with _TOKEN_USAGE_LOCK:
        print(
            "[LLM total token usage] "
            f"regular_input_tokens={_TOTAL_REGULAR_INPUT_TOKENS}, "
            f"cached_input_tokens={_TOTAL_CACHED_INPUT_TOKENS}, "
            f"total_output_tokens={_TOTAL_OUTPUT_TOKENS}",
            flush=True,
        )


def token_count(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def extract_response_text(payload: dict[str, Any]) -> str:
    status = payload.get("status")
    if status == "failed":
        raise LLMClientError(f"Model API returned an error payload: {payload.get('error')}")
    if status == "incomplete":
        raise LLMClientError(
            "Model API returned an incomplete response: "
            f"{payload.get('incomplete_details')}"
        )
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    output = payload.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if (
                    isinstance(part, dict)
                    and part.get("type") == "output_text"
                    and isinstance(part.get("text"), str)
                    and part["text"].strip()
                ):
                    return part["text"].strip()
    error = payload.get("error")
    if error:
        raise LLMClientError(f"Model API returned an error payload: {error}")
    raise LLMClientError(
        "Could not find output_text in the Responses API response "
        f"(keys={sorted(payload)})."
    )


def positive_int_env(name: str, default: int) -> int:
    value = nonnegative_int_env(name, default)
    if value < 1:
        raise LLMClientError(f"{name} must be at least 1, got: {value}")
    return value


def nonnegative_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise LLMClientError(f"{name} must be an integer, got: {raw}") from exc
    if value < 0:
        raise LLMClientError(f"{name} must be non-negative, got: {value}")
    return value
