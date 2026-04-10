from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.agent import KnowledgeBaseAgent
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore
from src.chunking import RecursiveChunker
from src.chunking import FixedSizeChunker
from src.chunking import SentenceChunker
from src.chunking import ChunkingStrategyComparator

SAMPLE_FILES = [
    # "data/python_intro.txt",
    # "data/vector_store_notes.md",
    # "data/rag_system_design.md",
    # "data/customer_support_playbook.txt",
    # "data/chunking_experiment_report.md",
    # "data/vi_retrieval_notes.md",
    "data/nau_an_formatted.txt"
]


def load_documents_from_files(file_paths: list[str]) -> list[Document]:
    """Load documents from file paths, CHUNK them, and return a list of Documents."""
    allowed_extensions = {".md", ".txt"}
    documents: list[Document] = []
    
    # Khởi tạo máy băm thịt (Chunker) với kích thước 500 ký tự
    chunker = RecursiveChunker(chunk_size=500)

    for raw_path in file_paths:
        path = Path(raw_path)

        if path.suffix.lower() not in allowed_extensions:
            print(f"Skipping unsupported file type: {path}")
            continue

        if not path.exists() or not path.is_file():
            print(f"Skipping missing file: {path}")
            continue

        # Đọc nguyên file
        full_text = path.read_text(encoding="utf-8")
        
        # Băm nhỏ file ra thành list các strings
        text_chunks = chunker.chunk(full_text)
        
        # Biến mỗi mẩu nhỏ thành một Document riêng biệt
        for i, chunk_content in enumerate(text_chunks):
            documents.append(
                Document(
                    # Tạo ID duy nhất cho từng mẩu (ví dụ: nau_an_formatted_0, nau_an_formatted_1)
                    id=f"{path.stem}_{i}",
                    content=chunk_content,
                    metadata={
                        "source": str(path), 
                        "extension": path.suffix.lower(),
                        "chunk_index": i
                    },
                )
            )

    return documents


def demo_llm(prompt: str) -> str:
    """A simple mock LLM for manual RAG testing."""
    return f"\n--- FULL PROMPT SENT TO AI ---\n{prompt}\n-----------------------------"


def run_manual_demo(question: str | None = None, sample_files: list[str] | None = None) -> int:
    files = sample_files or SAMPLE_FILES
    query = question or "Summarize the key information from the loaded files."

    print("=== Manual File Test ===")
    print("Accepted file types: .md, .txt")
    print("Input file list:")
    for file_path in files:
        print(f"  - {file_path}")

    docs = load_documents_from_files(files)
    if not docs:
        print("\nNo valid input files were loaded.")
        print("Create files matching the sample paths above, then rerun:")
        print("  python3 main.py")
        return 1

    print(f"\nLoaded {len(docs)} documents")
    for doc in docs:
        print(f"  - {doc.id}: {doc.metadata['source']}")

    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "local":
        try:
            embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    elif provider == "openai":
        try:
            embedder = OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    else:
        embedder = _mock_embed

    print(f"\nEmbedding backend: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")

    store = EmbeddingStore(collection_name="manual_test_store", embedding_fn=embedder)
    store.add_documents(docs)

    print(f"\nStored {store.get_collection_size()} documents in EmbeddingStore")
    print("\n=== EmbeddingStore Search Test ===")
    print(f"Query: {query}")
    search_results = store.search(query, top_k=3)
    for index, result in enumerate(search_results, start=1):
        print(f"{index}. score={result['score']:.3f} source={result['metadata'].get('source')}")
        print(f"   content preview: {result['content'][:120].replace(chr(10), ' ')}...")

    print("\n=== KnowledgeBaseAgent Test ===")
    agent = KnowledgeBaseAgent(store=store, llm_fn=demo_llm)
    print(f"Question: {query}")
    print("Agent answer:")
    print(agent.answer(query, top_k=3))
    return 0


def main() -> int:
    question = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else None
    return run_manual_demo(question=question)


if __name__ == "__main__":
    raise SystemExit(main())
