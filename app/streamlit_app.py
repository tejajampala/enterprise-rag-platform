"""Streamlit portfolio interface for Consumer Finance Intelligence."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from consumer_finance_rag.bedrock import BedrockConfigurationError
from consumer_finance_rag.config import load_settings
from consumer_finance_rag.factory import build_pipeline
from consumer_finance_rag.pipeline import RAGPipeline

st.set_page_config(
    page_title="Consumer Finance Intelligence",
    page_icon="🏦",
    layout="centered",
)

st.title("Consumer Finance Intelligence")
st.caption("An evidence-grounded RAG system using public U.S. consumer-finance information.")

SUGGESTIONS = [
    "What should I compare when choosing a credit card?",
    "How can I dispute an error on my credit report?",
    "What documents should I review before closing on a mortgage?",
    "What information must a debt collector give me?",
]


@st.cache_resource
def get_pipeline() -> RAGPipeline:
    return build_pipeline(load_settings())


with st.expander("Suggested questions", expanded=True):
    for suggestion in SUGGESTIONS:
        st.markdown(f"- {suggestion}")

question = st.text_area(
    "Ask a consumer-finance question",
    placeholder="For example: How do I dispute a credit-report error?",
    height=100,
)

if st.button("Ask", type="primary", use_container_width=True):
    if not question.strip():
        st.warning("Enter a question first.")
    elif not load_settings().models.generation.model:
        st.error(
            "AWS Bedrock generation is not configured. Set `BEDROCK_GENERATION_MODEL` "
            "to a model ID enabled in your AWS account, then restart the app."
        )
    elif not Path("data/index/bm25_state.json").exists():
        st.error("The local index is missing. Run `uv run python scripts/ingest.py` first.")
    else:
        try:
            with st.spinner("Retrieving and grounding an answer…"):
                pipeline = get_pipeline()
                response = pipeline.answer(question.strip())
            st.subheader("Answer")
            st.markdown(response.answer)
            st.subheader("Sources")
            for citation in response.citations:
                if citation.source_url:
                    st.markdown(
                        f"- **[{citation.citation_id}]** "
                        f"[{citation.document_title}]({citation.source_url}) — "
                        f"`{citation.chunk_id}`"
                    )
                else:
                    st.markdown(
                        f"- **[{citation.citation_id}]** {citation.document_title} — "
                        f"`{citation.chunk_id}`"
                    )

            with st.expander("How this answer was generated"):
                retrieval = response.retrieval_metadata
                models = response.model_metadata
                st.markdown(f"**Retrieval strategy:** {retrieval.strategy}")
                st.markdown(f"**Dense candidates:** {retrieval.dense_candidates}")
                st.markdown(f"**Sparse candidates:** {retrieval.sparse_candidates}")
                st.markdown(f"**Candidates reranked:** {retrieval.reranked}")
                st.markdown(f"**Final contexts:** {retrieval.final_contexts}")
                st.markdown(f"**Embedding model:** `{models.embedding_model}`")
                st.markdown(f"**Reranker model:** `{models.reranker_model}`")
                st.markdown(f"**Generation model:** `{models.generation_model}`")
        except BedrockConfigurationError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(
                "The answer could not be generated. Confirm Qdrant is running, the corpus "
                "is indexed, and AWS credentials/model access are configured. "
                f"Details: {exc}"
            )
