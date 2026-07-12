# -*- coding: utf-8 -*-
"""Shared OpenAI-compatible Chat Completions client."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RETRYABLE_STATUS_CODES = {408, 409, 425, 429}


class LLMClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChatCompletionClient:
    api_key: str
    url: str
    model: str
    timeout_seconds: int
    max_retries: int

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        max_completion_tokens: int,
        thinking_type: str | None = "adaptive",
    ) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": max_completion_tokens,
        }
        if thinking_type:
            body["thinking"] = {"type": thinking_type}

        payload = self._post_with_retries(body)
        return extract_chat_completion_text(payload)

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
                    raise LLMClientError("Chat Completions response must be a JSON object.")
                return payload
            except (requests.RequestException, ValueError, LLMClientError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(2**attempt)
                    continue
                response_body = ""
                if isinstance(exc, requests.RequestException) and exc.response is not None:
                    response_body = f"; response body: {exc.response.text[:1000]}"
                raise LLMClientError(f"Chat Completions request failed: {exc}{response_body}") from exc
        raise LLMClientError(f"Chat Completions request failed: {last_error}")


def create_client() -> ChatCompletionClient:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("API_KEY", "").strip()
    model = os.getenv("MODEL", "").strip()
    url = resolve_chat_completions_url()
    timeout_seconds = positive_int_env("MODEL_TIMEOUT_SECONDS", 600)
    max_retries = nonnegative_int_env("MODEL_MAX_RETRIES", 2)
    missing = [name for name, value in (("API_KEY", api_key), ("MODEL", model)) if not value]
    if missing:
        raise LLMClientError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". Please create .env from .env.example."
        )
    return ChatCompletionClient(api_key, url, model, timeout_seconds, max_retries)


def resolve_chat_completions_url() -> str:
    configured = os.getenv("MODEL_URL", "").strip()
    if configured:
        return configured

    return ""

def extract_chat_completion_text(payload: dict[str, Any]) -> str:
    base_resp = payload.get("base_resp")
    if isinstance(base_resp, dict) and base_resp.get("status_code", 0) != 0:
        raise LLMClientError(
            "Model API returned an error payload: "
            f"{base_resp.get('status_code')}: {base_resp.get('status_msg', '')}"
        )
    choices = payload.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
            text = choice.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()
    error = payload.get("error")
    if error:
        raise LLMClientError(f"Model API returned an error payload: {error}")
    raise LLMClientError(
        "Could not find choices[].message.content in the Chat Completions response "
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
