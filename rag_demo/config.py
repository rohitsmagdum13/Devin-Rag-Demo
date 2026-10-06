import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from rag_demo.errors import AppError


@dataclass(frozen=True)
class Settings:
    store_path: Path = Path("data/chroma")
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    answer_model: str = "gpt-4o-mini"
    chunk_size: int = 1200
    chunk_overlap: int = 200
    max_upload_mb: int = 20
    retrieval_k: int = 5
    max_retrieval_calls: int = 3
    timeout_seconds: int = 45

    def __post_init__(self) -> None:
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise AppError("CHUNK_OVERLAP must be nonnegative and smaller than CHUNK_SIZE.")
        if (
            min(
                self.embedding_dimensions,
                self.max_upload_mb,
                self.retrieval_k,
                self.max_retrieval_calls,
                self.timeout_seconds,
            )
            < 1
        ):
            raise AppError(
                "Dimensions, upload limit, retrieval limits, and timeout must be positive."
            )
        if self.max_retrieval_calls > 5:
            raise AppError("MAX_RETRIEVAL_CALLS must be at most 5 to bound cost and latency.")

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        try:
            return cls(
                store_path=Path(os.getenv("VECTOR_STORE_PATH", "data/chroma")),
                embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
                embedding_dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "1536")),
                answer_model=os.getenv("ANSWER_MODEL", "gpt-4o-mini"),
                chunk_size=int(os.getenv("CHUNK_SIZE", "1200")),
                chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
                max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "20")),
                retrieval_k=int(os.getenv("RETRIEVAL_K", "5")),
                max_retrieval_calls=int(os.getenv("MAX_RETRIEVAL_CALLS", "3")),
                timeout_seconds=int(os.getenv("OPENAI_TIMEOUT_SECONDS", "45")),
            )
        except ValueError:
            raise AppError("Numeric environment settings must contain whole numbers.") from None
