from dataclasses import dataclass, field


@dataclass(frozen=True)
class TextSection:
    text: str
    page: int | None = None


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    document_name: str
    text: str
    page: int | None = None


@dataclass
class Usage:
    embedding_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    requests: int = 0

    def add(self, other: "Usage") -> None:
        self.embedding_tokens += other.embedding_tokens
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.requests += other.requests


@dataclass(frozen=True)
class IngestionResult:
    document_id: str
    document_name: str
    chunk_count: int
    duplicate: bool
    latency_seconds: float
    usage: Usage = field(default_factory=Usage)
