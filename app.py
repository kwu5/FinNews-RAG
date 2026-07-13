"""Streamlit demo for FinNews-RAG grounded Q&A (Ship I polish).

THE demo: a query box over the chunk index that returns a grounded, source-cited
answer — and abstains instead of hallucinating when nothing relevant is retrieved.
Run with:  streamlit run app.py

Cold start loads the embedding model (~a few seconds) the first time a query runs
— that's the model load, not a hang.

Heavy singletons (EmbeddingGenerator / VectorStore / LLMClient / Settings) are
built once via @st.cache_resource; Streamlit re-runs this whole script top-to-
bottom on every interaction, so we must not reconstruct them per query.
"""

import streamlit as st

from src.config import Settings
from src.processing.embeddings import EmbeddingGenerator
from src.storage.vector_store import VectorStore
from src.summarization.llm_client import LLMClient
from src.rag.retriever import Retriever
from src.rag.qa import QAEngine


# Example questions that pre-fill the query box — chosen to land in-domain against
# the financial-news corpus so the demo answers well on first click.
EXAMPLE_QUERIES = [
    "What did the Fed signal on interest rates?",
    "Which companies reported earnings recently?",
    "What's the latest news on cryptocurrency?",
]


@st.cache_resource
def get_settings() -> Settings:
    """Build Settings once and cache it (item 7 — was built twice before)."""
    return Settings()  # type: ignore


@st.cache_resource
def get_engine() -> QAEngine:
    """Build the QA engine once and cache it across Streamlit re-runs."""
    settings = get_settings()
    embedder = EmbeddingGenerator(settings)
    vstore = VectorStore(settings)
    llm = LLMClient(settings)
    retriever = Retriever(embedder, vstore)
    return QAEngine(retriever, llm)


def render_sidebar() -> None:
    """About panel — ties the demo to the README's 'grounded + cited + evaluated' story."""
    with st.sidebar:
        st.header("About")
        st.markdown(
            "**FinNews-RAG** answers questions about recent financial news with "
            "answers grounded in retrieved articles and cited by source.\n\n"
            "- Retrieval-augmented Q&A over a chunk-level ChromaDB index\n"
            "- Abstains instead of hallucinating when nothing relevant is found\n"
            "- **Custom** evaluation harness (no LangChain / LlamaIndex): "
            "faithfulness **0.96**, answer-relevance **0.82**\n\n"
            "Embeddings: `all-MiniLM-L6-v2` · Generation: `gpt-4o-mini`"
        )


def render_examples() -> None:
    """Example-query buttons. Must render BEFORE the form's text_input so that a
    click can write session_state['query_box'] before that widget is instantiated
    (Streamlit forbids mutating a widget-backed key after the widget exists)."""
    st.caption("Try an example:")
    cols = st.columns(len(EXAMPLE_QUERIES))
    for col, example in zip(cols, EXAMPLE_QUERIES):
        # Unique key per button; on click, pre-fill the (not-yet-created) input.
        if col.button(example, key=f"ex_{example}", use_container_width=True):
            st.session_state["query_box"] = example


def render_sources(citations) -> None:
    """One bordered card per cited source, with a clickable headline link."""
    st.subheader("Sources")
    for c in citations:
        with st.container(border=True):
            st.markdown(
                f"**[{c.title}]({c.url})**  \n{c.source} · cited as [{c.marker}]"
            )


def main() -> None:
    # set_page_config MUST be the first Streamlit call in the run.
    st.set_page_config(page_title="FinNews-RAG", page_icon="📈", layout="wide")

    settings = get_settings()
    engine = get_engine()

    render_sidebar()

    st.title("📈 FinNews-RAG")
    st.caption(
        "Ask about recent financial news — answers are grounded in retrieved "
        "articles and cited by source."
    )

    # Examples go ABOVE the form (plain buttons can't live inside st.form).
    render_examples()

    # The form: Enter-in-field or the submit button triggers one rerun on submit.
    with st.form("ask"):
        query = st.text_input(
            "Ask a question about recent financial news", key="query_box"
        )
        submitted = st.form_submit_button("Ask")

    # Answer when the form was submitted OR an example pre-filled the box — but only
    # if there's real text to answer.
    if not (submitted or query.strip()):
        return
    if not query.strip():
        return

    with st.spinner("Retrieving + answering…"):
        result = engine.answer_query(query, settings.RETRIEVAL_TOP_K)

    # Abstention banner first — covers both the zero-hit case and the LLM-can't-
    # answer-from-context case (both set answered_from_context=False).
    if not result.answered_from_context:
        st.warning("Couldn't answer this confidently from the indexed news.")

    st.markdown(result.answer)

    if result.citations:
        render_sources(result.citations)


if __name__ == "__main__":
    main()
