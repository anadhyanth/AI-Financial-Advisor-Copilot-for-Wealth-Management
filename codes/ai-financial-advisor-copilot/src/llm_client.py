"""
Thin wrapper around the Anthropic Messages API so the rest of the codebase
never talks to the SDK directly. Keeping this isolated makes it trivial to
swap providers, add retries/caching, or mock in tests.
"""

from typing import List, Optional

import anthropic

import config
from src.utils import get_logger

logger = get_logger(__name__)

_client: Optional[anthropic.Anthropic] = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        if not config.ANTHROPIC_API_KEY:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Add it to your environment or .env file."
            )
        _client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _client


def generate_response(
    system_prompt: str,
    user_message: str,
    conversation_history: Optional[List[dict]] = None,
) -> str:
    """
    Send a system prompt + user message (with optional prior turns) to the LLM
    and return the assistant's text response.

    conversation_history, if provided, should be a list of
    {"role": "user"|"assistant", "content": str} dicts, oldest first.
    """
    client = get_client()
    messages = list(conversation_history or [])
    messages.append({"role": "user", "content": user_message})

    try:
        response = client.messages.create(
            model=config.LLM_MODEL,
            max_tokens=config.LLM_MAX_TOKENS,
            temperature=config.LLM_TEMPERATURE,
            system=system_prompt,
            messages=messages,
        )
    except anthropic.APIError as e:
        logger.error("LLM call failed: %s", e)
        raise RuntimeError(f"LLM request failed: {e}") from e

    text_blocks = [block.text for block in response.content if block.type == "text"]
    return "\n".join(text_blocks).strip()
