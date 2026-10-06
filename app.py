import hashlib
import os

import streamlit as st

from rag_demo.agent import RetrievalAgent
from rag_demo.config import Settings
from rag_demo.errors import AppError
from rag_demo.extraction import extract
from rag_demo.ingestion import ingest
from rag_demo.models import AnswerResult
from rag_demo.providers import OpenAIProvider
from rag_demo.store import VectorStore


@st.cache_resource
def open_store(path: str, model: str, dimensions: int) -> VectorStore:
    from pathlib import Path

    return VectorStore(Path(path), model, dimensions)


def show_answer(result: AnswerResult, turn: int) -> None:
    st.text(result.answer)
    st.caption(
        f"Response time: {result.latency_seconds:.2f}s · "
        f"Retrieval calls: {result.retrieval_calls} · "
        f"Tokens: {result.usage.prompt_tokens + result.usage.completion_tokens} LLM / "
        f"{result.usage.embedding_tokens} embedding"
    )
    for number, citation in enumerate(result.citations, start=1):
        chunk = citation.chunk
        anchor = f"source-{turn}-{hashlib.sha256(chunk.id.encode()).hexdigest()}-{number}"
        st.markdown(f"[Source {number}](#{anchor})")
        st.html(f'<div id="{anchor}"></div>')
        page = f" · page {chunk.page}" if chunk.page is not None else ""
        with st.expander(f"Source {number}: {chunk.document_name}{page}"):
            st.caption(f"Chunk ID: {chunk.id}")
            st.text(citation.quote)
            st.text(chunk.text)
    with st.expander("Retrieval details"):
        st.write(
            {
                "mode": result.mode,
                "queries": result.queries,
                "retrieved_chunks": len(result.retrieved_chunks),
            }
        )


def main() -> None:
    st.set_page_config(page_title="Document Q&A", page_icon=None, layout="wide")
    st.title("Ask your documents")
    st.caption("Agentic retrieval · OpenAI · persistent local Chroma")
    st.write("Upload documents, ask a question, and inspect the evidence behind each answer.")
    try:
        settings = Settings.from_env()
        store = open_store(
            str(settings.store_path), settings.embedding_model, settings.embedding_dimensions
        )
    except AppError as error:
        st.error(str(error))
        st.stop()
    except Exception:
        st.error("Cannot open the local index. Check VECTOR_STORE_PATH and disk permissions.")
        st.stop()

    if (
        not os.getenv("OPENAI_API_KEY", "").strip()
        or os.getenv("OPENAI_API_KEY") == "replace-with-your-key"
    ):
        st.info("Set OPENAI_API_KEY in your environment or local .env to enable indexing and Q&A.")
    with st.sidebar:
        st.header("Documents")
        st.caption("PDF, DOCX, UTF-8 TXT · " + str(settings.max_upload_mb) + " MB per file")
        transport_limit = int(st.get_option("server.maxUploadSize"))
        if settings.max_upload_mb != transport_limit:
            st.caption(
                f"Application validation: {settings.max_upload_mb} MB; "
                f"uploader transport limit: {transport_limit} MB. The lower limit applies."
            )
        uploads = st.file_uploader(
            "Upload documents", type=["pdf", "docx", "txt"], accept_multiple_files=True
        )
        st.caption(
            "Document text is sent to OpenAI. Only upload material you have permission to share."
        )
        if st.button("Index documents", disabled=not uploads, type="primary") and uploads:
            progress = st.progress(0.0, text="Validating documents")
            for index, upload in enumerate(uploads):
                st.text(upload.name)
                try:
                    content = upload.getvalue()
                    extract(upload.name, content, settings.max_upload_mb * 1024 * 1024)
                    with st.status(
                        "Extracting → chunking → embedding → saving", expanded=True
                    ) as status:
                        result = ingest(
                            upload.name, content, settings, store, OpenAIProvider(settings)
                        )
                        if result.duplicate:
                            st.success(f"Already indexed · {result.chunk_count} chunks (duplicate)")
                        else:
                            st.success(
                                f"Indexed · {result.chunk_count} chunks · "
                                f"{result.latency_seconds:.2f}s"
                            )
                        status.update(label="Document ready", state="complete", expanded=True)
                except AppError as error:
                    st.error(str(error))
                except Exception:
                    st.error("Indexing failed. Check local disk permissions and retry.")
                progress.progress((index + 1) / len(uploads), text="Processing documents")
            progress.progress(1.0, text="Upload processing complete")
        st.subheader("Indexed documents")
        documents = store.documents()
        if documents:
            st.dataframe(
                [{"Document": d["name"], "Chunks": d["chunks"]} for d in documents],
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("No documents yet. Upload and index a file to begin.")
        st.caption(f"{len(documents)} documents · {store.count()} chunks")
        if st.button("Clear conversation"):
            st.session_state.turns = []
            st.rerun()

    if "turns" not in st.session_state:
        st.session_state.turns = []
    st.subheader("Conversation")
    if not st.session_state.turns:
        st.info("Try: What is the reimbursement deadline?")
    for turn, (question, answer) in enumerate(st.session_state.turns):
        with st.chat_message("user"):
            st.text(question)
        with st.chat_message("assistant"):
            show_answer(answer, turn)
    question = st.chat_input("Ask about your documents")
    if question:
        try:
            history = [(q, answer.answer) for q, answer in st.session_state.turns]
            with st.spinner("Planning retrieval and checking the evidence…"):
                answer = RetrievalAgent(settings, store, OpenAIProvider(settings)).ask(
                    question, history
                )
            st.session_state.turns.append((question, answer))
            st.rerun()
        except AppError as error:
            st.error(str(error))
        except Exception:
            st.error("Could not answer the question. Check your local index and try again.")
    st.caption(
        "Answers require document evidence. Source quote checks are deterministic; "
        "model grounding is not guaranteed. Chat history lasts only for this browser session."
    )


if __name__ == "__main__":
    main()
