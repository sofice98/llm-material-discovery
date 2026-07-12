"""Shared client for calls to the configured chat-completions provider."""

from .chat_client import ChatCompletionClient, LLMClientError, create_client

__all__ = ["ChatCompletionClient", "LLMClientError", "create_client"]
