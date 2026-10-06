import math
from collections import Counter
from pathlib import Path

import chromadb
import numpy as np
from chromadb.api.types import IncludeEnum
from chromadb.config import Settings as ChromaSettings

from rag_demo.errors import AppError
from rag_demo.models import Chunk


class VectorStore:
    def __init__(self, path: Path, model: str, dimensions: int) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.dimensions = dimensions
        self.client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            "document_chunks",
            embedding_function=None,
            metadata={"hnsw:space": "cosine", "embedding_model": model, "dimensions": dimensions},
        )
        metadata = self.collection.metadata or {}
        if metadata.get("embedding_model") != model or metadata.get("dimensions") != dimensions:
            raise AppError(
                "Index model/dimensions differ. Use a new VECTOR_STORE_PATH or original settings."
            )

    def document_chunks(self, document_id: str) -> list[str]:
        return self.collection.get(where={"document_id": document_id}, include=[])["ids"]

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings) or not chunks:
            raise AppError("Embedding count mismatch; no document was saved.")
        if any(
            len(vector) != self.dimensions or not all(math.isfinite(x) for x in vector)
            for vector in embeddings
        ):
            raise AppError("Embedding dimensions or values are invalid; no document was saved.")
        ids = [chunk.id for chunk in chunks]
        try:
            for start in range(0, len(chunks), 500):
                batch = chunks[start : start + 500]
                self.collection.add(
                    ids=[chunk.id for chunk in batch],
                    embeddings=np.asarray(embeddings[start : start + 500], dtype=np.float32),
                    documents=[chunk.text for chunk in batch],
                    metadatas=[
                        {
                            "document_id": chunk.document_id,
                            "document_name": chunk.document_name,
                            "page": chunk.page or 0,
                        }
                        for chunk in batch
                    ],
                )
        except Exception:
            self.collection.delete(ids=ids)
            raise AppError(
                "Could not save the document. Check local disk space and retry."
            ) from None

    def documents(self) -> list[dict[str, str | int]]:
        result = self.collection.get(include=[IncludeEnum.metadatas])
        metadata = result["metadatas"] or []
        counts = Counter(str(item["document_id"]) for item in metadata)
        names = {str(item["document_id"]): str(item["document_name"]) for item in metadata}
        return [
            {"id": doc_id, "name": names[doc_id], "chunks": count}
            for doc_id, count in sorted(counts.items())
        ]

    def count(self) -> int:
        return self.collection.count()

    def search(self, embedding: list[float], k: int) -> list[Chunk]:
        if not self.count():
            return []
        result = self.collection.query(
            query_embeddings=np.asarray([embedding], dtype=np.float32),
            n_results=min(k, self.count()),
            include=[IncludeEnum.documents, IncludeEnum.metadatas],
        )
        texts = (result["documents"] or [[]])[0]
        metadata = (result["metadatas"] or [[]])[0]
        return [
            Chunk(
                chunk_id,
                str(meta["document_id"]),
                str(meta["document_name"]),
                text or "",
                int(meta["page"]) or None,
            )
            for chunk_id, text, meta in zip(result["ids"][0], texts, metadata, strict=True)
        ]
