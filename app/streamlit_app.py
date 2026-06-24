"""Streamlit UI for ClauseFinder grounded regulation Q and A."""

from __future__ import annotations

import streamlit as st

from clausefinder import config
from clausefinder.process import normalize
from clausefinder.rag.generate import Answer, answer_question, generate_answer
from clausefinder.rag.prompt import REFUSAL_MESSAGE
from clausefinder.rag.retrieve import Retriever, get_retriever, is_low_confidence

EXAMPLE_QUESTIONS = [
    "What clear opening width is required for an accessible entrance door?",
    "What is the minimum guarding height for stairs in a dwelling?",
    "What is the maximum travel distance where escape is possible in one direction only?",
    "What U-value is required for external walls?",
]
FETCH_MULTIPLIER = 4

st.set_page_config(
    page_title="ClauseFinder — Building Regulation Q&A",
    page_icon="📑",
    layout="centered",
)


@st.cache_resource
def load_retriever() -> Retriever:
    """Load and cache the FAISS retriever once per app process."""
    return get_retriever()


@st.cache_data(show_spinner=False)
def list_source_options() -> list[tuple[str, str]]:
    """Return display-label and source-id pairs from normalized documents."""
    try:
        conn = normalize.connect(config.SQLITE_DB_PATH)
        try:
            rows = conn.execute(
                """
				SELECT DISTINCT source, title
				FROM documents
				WHERE source IS NOT NULL
				ORDER BY title
				"""
            ).fetchall()
            return [(str(row["title"] or row["source"]), str(row["source"])) for row in rows]
        finally:
            conn.close()
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def run_query(query: str, selected_sources: tuple[str, ...], k: int) -> Answer:
    """Run one grounded query, optionally restricted to selected sources."""
    if not selected_sources:
        return answer_question(query, k=k)

    retriever = load_retriever()
    chunks = retriever.search(query, k=max(k * FETCH_MULTIPLIER, 20))
    chunks = [chunk for chunk in chunks if chunk.source in selected_sources][:k]

    if not chunks:
        return Answer(
            query=query,
            answer=REFUSAL_MESSAGE,
            sources=[],
            low_confidence=True,
            refused=True,
            model=config.GEMINI_MODEL,
        )

    text = generate_answer(query, chunks)
    return Answer(
        query=query,
        answer=text,
        sources=chunks,
        low_confidence=is_low_confidence(chunks),
        refused=text.strip().startswith(REFUSAL_MESSAGE),
        model=config.GEMINI_MODEL,
    )


def set_example(question: str) -> None:
    """Copy an example question into the input and trigger execution."""
    st.session_state.query = question
    st.session_state.trigger = True


def trigger_ask() -> None:
    """Trigger execution for the current input query."""
    st.session_state.trigger = True


st.title("ClauseFinder")
st.caption("Ask grounded questions about indexed building-regulation sources.")

try:
    load_retriever()
except Exception:
    st.error("Index or database not found. Run: uv run python scripts/build_all.py")
    st.stop()

st.sidebar.title("ClauseFinder")
st.sidebar.write("Grounded question answering over indexed building regulations.")
st.sidebar.markdown(
    "**Sources**\n"
    "UK Building Regulations - Approved Documents (Parts B, K, M) and the "
    "Building Regulations 2010, all under the Open Government Licence v3.0."
)
st.sidebar.markdown(
    "**Limitations**\n"
    "Not legal advice; answers come ONLY from the indexed England documents; "
    "the assistant refuses when the answer is not in its sources; always confirm "
    "against the cited original."
)
st.sidebar.subheader("Settings")
k = st.sidebar.slider("Passages to retrieve (k)", 3, 10, value=config.TOP_K)

source_options = list_source_options()
label_by_value = {value: label for label, value in source_options}
selected_sources = st.sidebar.multiselect(
    "Filter by document",
    options=[value for _, value in source_options],
    format_func=lambda value: label_by_value.get(value, value),
)
st.sidebar.caption(f"Model: {config.GEMINI_MODEL} | Deterministic (temperature 0)")

st.session_state.setdefault("query", "")
st.session_state.setdefault("trigger", False)

columns = st.columns(2)
for i, question in enumerate(EXAMPLE_QUESTIONS):
    with columns[i % 2]:
        st.button(
            question,
            key=f"ex_{i}",
            on_click=set_example,
            args=(question,),
            use_container_width=True,
        )

st.text_input(
    "Your question",
    key="query",
    placeholder="Ask about a building regulation…",
)
st.button("Ask", type="primary", on_click=trigger_ask)

if st.session_state.trigger and st.session_state.query.strip():
    st.session_state.trigger = False
    try:
        with st.spinner("Searching the regulations…"):
            st.session_state.result = run_query(
                st.session_state.query,
                tuple(selected_sources),
                k,
            )
    except Exception as exc:
        st.session_state.result = None
        st.error(f"Generation failed: {exc}. Check your GEMINI_API_KEY in .env and your network.")

result = st.session_state.get("result")

if result is not None and result.query == st.session_state.query:
    st.subheader("Answer")
    if result.refused:
        st.info(result.answer)
    else:
        if result.low_confidence and result.sources:
            top_score = result.sources[0].score
            st.warning(
                "Low retrieval confidence "
                f"(top similarity {top_score:.2f}). "
                "Treat with caution and check the cited sources below."
            )
        st.markdown(result.answer)

    st.subheader(f"Sources ({len(result.sources)})")
    if result.sources:
        for i, chunk in enumerate(result.sources, 1):
            with st.expander(
                f"[{i}] {chunk.title} — {chunk.section}  ·  similarity {chunk.score:.2f}"
            ):
                if chunk.url:
                    st.markdown(f"[Open source document]({chunk.url})")
                st.markdown(chunk.text)
    else:
        st.caption("No passages were retrieved for this query.")
