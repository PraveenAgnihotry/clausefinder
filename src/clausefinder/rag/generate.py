"""Grounded generation and retrieval orchestration for ClauseFinder RAG."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from google import genai
from google.genai import types

from clausefinder import config
from clausefinder.rag import prompt, retrieve
from clausefinder.rag.retrieve import RetrievedChunk

_CLIENT: genai.Client | None = None


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

    resp = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=prompt.SYSTEM_INSTRUCTION,
            temperature=config.GEMINI_TEMPERATURE,
            max_output_tokens=config.GEMINI_MAX_OUTPUT_TOKENS,
        ),
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
    refused = text.strip().startswith(prompt.REFUSAL_MESSAGE) or (low_conf and not chunks)

    return Answer(
        query=query,
        answer=text,
        sources=chunks,
        low_confidence=low_conf,
        refused=refused,
        model=config.GEMINI_MODEL,
    )
