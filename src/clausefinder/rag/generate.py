"""Grounded generation and retrieval orchestration for ClauseFinder RAG."""

from __future__ import annotations

import os
import ssl
import time
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from clausefinder import config
from clausefinder.rag import prompt, retrieve
from clausefinder.rag.retrieve import RetrievedChunk

_CLIENT: genai.Client | None = None
_TRANSIENT_RETRY_CODES = {429, 500, 503}
_MAX_GENERATE_ATTEMPTS = 3


def get_client() -> genai.Client:
    """Return a lazy singleton Gemini client configured from environment."""
    global _CLIENT
    if _CLIENT is None:
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not found. Copy .env.example to .env and add your key."
            )
        _CLIENT = genai.Client(api_key=api_key)
    return _CLIENT


@dataclass(frozen=True, slots=True)
class Answer:
    """Structured answer payload for grounded question answering."""

    query: str
    answer: str
    sources: list[RetrievedChunk]
    low_confidence: bool
    refused: bool
    model: str


def generate_answer(query: str, chunks: list[RetrievedChunk]) -> str:
    """Generate a grounded answer text from retrieved context passages."""
    user_prompt = prompt.build_user_prompt(query, chunks)
    client = get_client()

    for attempt in range(1, _MAX_GENERATE_ATTEMPTS + 1):
        try:
            resp = client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=prompt.SYSTEM_INSTRUCTION,
                    temperature=config.GEMINI_TEMPERATURE,
                    max_output_tokens=config.GEMINI_MAX_OUTPUT_TOKENS,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            break
        except errors.APIError as exc:
            if exc.code not in _TRANSIENT_RETRY_CODES or attempt == _MAX_GENERATE_ATTEMPTS:
                raise
            time.sleep(2 ** (attempt - 1))
        except (httpx.TransportError, ssl.SSLError):
            if attempt == _MAX_GENERATE_ATTEMPTS:
                raise
            time.sleep(2 ** (attempt - 1))

    candidate = resp.candidates[0] if resp.candidates else None
    finish_reason = getattr(candidate, "finish_reason", None)
    if finish_reason == types.FinishReason.MAX_TOKENS:
        raise RuntimeError(
            "Gemini response was truncated at max_output_tokens; refusing to return an "
            "incomplete compliance answer."
        )

    answer_text = resp.text
    if not answer_text:
        return prompt.REFUSAL_MESSAGE
    return answer_text


def answer_question(
    query: str,
    *,
    k: int | None = None,
    retriever: retrieve.Retriever | None = None,
) -> Answer:
    """Retrieve relevant passages and produce a grounded answer."""
    active_retriever = retriever or retrieve.get_retriever()
    chunks = active_retriever.search(query, k)

    if not chunks:
        return Answer(
            query=query,
            answer=prompt.REFUSAL_MESSAGE,
            sources=[],
            low_confidence=True,
            refused=True,
            model=config.GEMINI_MODEL,
        )

    low_conf = retrieve.is_low_confidence(chunks)
    text = generate_answer(query, chunks)
    refused = text.strip().startswith(prompt.REFUSAL_MESSAGE)

    return Answer(
        query=query,
        answer=text,
        sources=chunks,
        low_confidence=low_conf,
        refused=refused,
        model=config.GEMINI_MODEL,
    )
