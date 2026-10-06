from rag_demo.errors import AppError
from rag_demo.models import Chunk, TextSection


def chunk_sections(
    sections: list[TextSection], document_id: str, name: str, size: int, overlap: int
) -> list[Chunk]:
    if not 0 <= overlap < size:
        raise AppError("Chunk overlap must be nonnegative and less than chunk size.")
    chunks: list[Chunk] = []
    for section in sections:
        for start in range(0, len(section.text), size - overlap):
            text = section.text[start : start + size].strip()
            if text:
                chunks.append(
                    Chunk(f"{document_id}:{len(chunks)}", document_id, name, text, section.page)
                )
            if start + size >= len(section.text):
                break
    return chunks
