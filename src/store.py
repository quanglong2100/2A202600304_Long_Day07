from __future__ import annotations

import uuid
from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] =[]
        self._collection = None
        self._next_index = 0

        try:
            import chromadb

            # Initialize an ephemeral client and get/create the collection
            client = chromadb.Client()
            self._collection = client.get_or_create_collection(name=self._collection_name)
            self._use_chroma = True
        except ImportError:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        """Build a normalized stored record for one document."""
        # Assume Document object has `content` and `metadata`. Fallback safely.
        content = getattr(doc, "content", str(doc))
        metadata = getattr(doc, "metadata", {})
        doc_id = getattr(doc, "id", str(uuid.uuid4()))
        
        return {
            "id": doc_id,
            "content": content,
            "metadata": metadata,
            "embedding": self._embedding_fn(content)
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """Run in-memory similarity search over provided records."""
        if not records:
            return[]
            
        query_emb = self._embedding_fn(query)
        scored_records =[]
        
        for rec in records:
            # Calculate dot product as similarity score
            score = _dot(query_emb, rec["embedding"])
            scored_records.append((score, rec))
            
        # Sort by score in descending order
        scored_records.sort(key=lambda x: x[0], reverse=True)
        
        # Return only the top_k records without the internal score wrapper
        return [rec for score, rec in scored_records[:top_k]]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        if not docs:
            return

        if self._use_chroma:
            ids = []
            documents =[]
            embeddings = []
            metadatas =[]
            
            for doc in docs:
                content = getattr(doc, "content", str(doc))
                doc_id = getattr(doc, "id", str(uuid.uuid4()))
                meta = getattr(doc, "metadata", {})
                
                ids.append(doc_id)
                documents.append(content)
                embeddings.append(self._embedding_fn(content))
                metadatas.append(meta)
                
            self._collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
        else:
            for doc in docs:
                self._store.append(self._make_record(doc))

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        if self._use_chroma:
            query_emb = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_emb],
                n_results=top_k
            )
            
            formatted_results =[]
            # Chroma returns lists of lists (for batch queries). We queried 1 text, so we access index [0]
            if results and results.get("ids") and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    formatted_results.append({
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i] if results.get("metadatas") else {}
                    })
            return formatted_results
        else:
            return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        metadata_filter = metadata_filter or {}
        
        if self._use_chroma:
            query_emb = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_emb],
                n_results=top_k,
                where=metadata_filter # ChromaDB directly supports 'where' filters
            )
            
            formatted_results =[]
            if results and results.get("ids") and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    formatted_results.append({
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i] if results.get("metadatas") else {}
                    })
            return formatted_results
        else:
            # Fallback in-memory behavior
            filtered_records = self._store
            if metadata_filter:
                filtered_records =[]
                for rec in self._store:
                    rec_meta = rec.get("metadata", {})
                    # Add to list only if all key-value pairs in the filter match the record's metadata
                    if all(rec_meta.get(k) == v for k, v in metadata_filter.items()):
                        filtered_records.append(rec)
                        
            return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        if self._use_chroma:
            initial_count = self._collection.count()
            self._collection.delete(where={"doc_id": doc_id})
            # If the count changed, we successfully deleted at least one chunk
            return self._collection.count() < initial_count
        else:
            initial_len = len(self._store)
            # Retain only chunks that DO NOT have matching doc_id
            self._store =[rec for rec in self._store if rec.get("metadata", {}).get("doc_id") != doc_id]
            return len(self._store) < initial_len
