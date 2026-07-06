from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

import requests


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "25"))
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
OLLAMA_WARMUP_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_WARMUP_TIMEOUT_SECONDS", "45"))

def _ollama_chat(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.1,
    timeout: float = OLLAMA_TIMEOUT_SECONDS,
) -> str | None:
    """Call the local Ollama chat endpoint and return text, or None if unavailable."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
        },
    }

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError, KeyError):
        return None

    message = data.get("message", {})
    content = message.get("content")
    return str(content).strip() if content else None


def _ollama_chat_stream(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.1,
    timeout: float = OLLAMA_TIMEOUT_SECONDS,
) -> Iterator[str]:
    """Call the local Ollama chat endpoint and yield generated text chunks."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": True,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
        },
    }

    try:
        with requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=timeout,
            stream=True,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except ValueError:
                    continue

                message = data.get("message", {})
                content = message.get("content")
                if content:
                    yield str(content)

                if data.get("done"):
                    break
    except requests.RequestException:
        return


def warm_up_ollama_model() -> bool:
    """Load the local Ollama model once so the first real chat turn is faster."""
    result = _ollama_chat(
        [
            {
                "role": "system",
                "content": "You are a local warm-up request. Reply with one short word.",
            },
            {"role": "user", "content": "ready"},
        ],
        temperature=0.0,
        timeout=OLLAMA_WARMUP_TIMEOUT_SECONDS,
    )
    return result is not None


def _polish_messages(
    *,
    question: str,
    base_answer: str,
    retrieved_context: str,
    trip_context: dict[str, Any],
    has_recommendation_cards: bool = False,
) -> list[dict[str, str]]:
    cards_instruction = (
        "The UI will display recommendation cards after your text. Do not repeat every listing; introduce the cards and explain how to read them."
        if has_recommendation_cards
        else "No recommendation cards are being displayed; answer directly using the grounded answer."
    )

    system = f"""
You are a friendly Airbnb holiday-planning assistant for Madrid and Tokyo.
Rewrite the grounded answer so it feels conversational, helpful, and lightly enthusiastic.

Hard rules:
- Do not add new listings, neighbourhoods, prices, ratings, model metrics, dates, or claims.
- Preserve all numbers and caveats from the grounded answer.
- If information is snapshot-based or not live, keep that limitation clear.
- Keep the answer concise and practical.
- Use markdown naturally.
- {cards_instruction}
""".strip()

    user = json.dumps(
        {
            "user_question": question,
            "trip_context": trip_context,
            "grounded_answer": base_answer,
            "retrieved_project_context": retrieved_context[:1800],
        },
        default=str,
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def polish_answer_with_llm(
    *,
    question: str,
    base_answer: str,
    retrieved_context: str,
    trip_context: dict[str, Any],
    has_recommendation_cards: bool = False,
) -> str:
    """Use Qwen/Ollama to make a grounded answer friendlier without changing facts."""
    if not base_answer.strip():
        return base_answer

    polished = _ollama_chat(
        _polish_messages(
            question=question,
            base_answer=base_answer,
            retrieved_context=retrieved_context,
            trip_context=trip_context,
            has_recommendation_cards=has_recommendation_cards,
        ),
        temperature=0.35,
        timeout=OLLAMA_TIMEOUT_SECONDS,
    )
    return polished.strip() if polished else base_answer


def stream_polished_answer_with_llm(
    *,
    question: str,
    base_answer: str,
    retrieved_context: str,
    trip_context: dict[str, Any],
    has_recommendation_cards: bool = False,
) -> Iterator[str]:
    """Stream Qwen/Ollama polish chunks, falling back to the grounded answer."""
    if not base_answer.strip():
        yield base_answer
        return

    yielded_any = False
    for chunk in _ollama_chat_stream(
        _polish_messages(
            question=question,
            base_answer=base_answer,
            retrieved_context=retrieved_context,
            trip_context=trip_context,
            has_recommendation_cards=has_recommendation_cards,
        ),
        temperature=0.35,
        timeout=OLLAMA_TIMEOUT_SECONDS,
    ):
        yielded_any = True
        yield chunk

    if not yielded_any:
        yield base_answer
