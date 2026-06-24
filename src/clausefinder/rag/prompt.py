"""Build grounded RAG prompt strings from retrieved context passages."""

from __future__ import annotations

from clausefinder.rag.retrieve import RetrievedChunk

REFUSAL_MESSAGE = "I could not find an answer to this question in the provided documents."

SYSTEM_INSTRUCTION = "\n".join(
    [
        "You answer questions about building regulations using ONLY the numbered "
        "context passages provided.",
        "Do not use any outside or prior knowledge. If the passages do not contain "
        "the answer, reply with exactly this sentence and nothing else: "
        f"{REFUSAL_MESSAGE}",
        "Support every factual statement with a citation to the passage number(s) it "
        "came from, written like [1] or [2][3]. Put the citation right after the "
        "statement it supports.",
        "Prefer quoting the clause or section number and the document name when they "
        "appear in a passage.",
        "Be concise and precise. Do not speculate, generalise, or fill gaps.",
        "Never invent passage numbers, clause numbers, document names, or URLs. "
        "Only cite passages that are actually shown.",
    ]
)


def _context_header(chunk: RetrievedChunk, number: int) -> str:
    """Build one citable header line for a context chunk."""
    header = f"[{number}] {chunk.source}"

    section = chunk.section.strip()
    url = chunk.url.strip()

    if section:
        header = f"{header} \u2014 {section}"
    if url:
        header = f"{header} ({url})"
    return header


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as numbered citable blocks."""
    blocks: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        header = _context_header(chunk, i)
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n".join(blocks)


def build_user_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    """Build the user prompt containing context, question, and grounding reminder."""
    context_text = format_context(chunks)
    reminder = (
        "Answer only from the passages above, cite every factual claim with [n], "
        f"and use this exact sentence if unsupported: {REFUSAL_MESSAGE}"
    )

    return f"Context passages:\n{context_text}\n\nQuestion:\n{query}\n\n{reminder}"
