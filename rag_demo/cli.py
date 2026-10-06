import argparse
import json
from dataclasses import asdict
from pathlib import Path

from rag_demo.config import Settings
from rag_demo.errors import AppError
from rag_demo.ingestion import ingest
from rag_demo.providers import OpenAIProvider
from rag_demo.store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Index local PDF, DOCX, and TXT documents")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest").add_argument("path", type=Path)
    sub.add_parser("list")
    args = parser.parse_args()
    try:
        settings = Settings.from_env()
        store = VectorStore(
            settings.store_path, settings.embedding_model, settings.embedding_dimensions
        )
        if args.command == "list":
            print(json.dumps(store.documents(), indent=2))
        else:
            result = ingest(
                args.path.name, args.path.read_bytes(), settings, store, OpenAIProvider(settings)
            )
            print(json.dumps(asdict(result), indent=2))
    except AppError as error:
        parser.exit(1, f"Error: {error}\n")
    except OSError:
        parser.exit(
            1, "Error: Could not read file or access local vector store. Check permissions.\n"
        )


if __name__ == "__main__":
    main()
