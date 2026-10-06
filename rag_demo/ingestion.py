import hashlib
from pathlib import Path
from time import perf_counter

from filelock import FileLock, Timeout

from rag_demo.chunking import chunk_sections
from rag_demo.config import Settings
from rag_demo.errors import AppError
from rag_demo.extraction import extract
from rag_demo.models import IngestionResult
from rag_demo.providers import Embedder
from rag_demo.store import VectorStore


def ingest(
    name: str, content: bytes, settings: Settings, store: VectorStore, embedder: Embedder
) -> IngestionResult:
    start = perf_counter()
    name = Path(name).name
    sections = extract(name, content, settings.max_upload_mb * 1024 * 1024)
    document_id = hashlib.sha256(content).hexdigest()
    try:
        with FileLock(str(store.path / "ingestion.lock"), timeout=60):
            existing = store.document_chunks(document_id)
            if existing:
                return IngestionResult(
                    document_id, name, len(existing), True, perf_counter() - start
                )
            chunks = chunk_sections(
                sections, document_id, name, settings.chunk_size, settings.chunk_overlap
            )
            embeddings, usage = embedder.embed([chunk.text for chunk in chunks])
            store.add(chunks, embeddings)
    except Timeout:
        raise AppError("Another document is being indexed. Wait a moment and retry.") from None
    return IngestionResult(document_id, name, len(chunks), False, perf_counter() - start, usage)
