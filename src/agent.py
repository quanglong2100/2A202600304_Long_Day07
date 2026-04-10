from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        # 1. Retrieve top-k relevant chunks from the store
        retrieved_records = self.store.search(query=question, top_k=top_k)
        
        # Extract content from the retrieved records
        # Assuming the records returned by EmbeddingStore are dicts with a 'content' key
        context_chunks = [record["content"] for record in retrieved_records]
        
        # Formulate a structured context string
        context_str = "\n\n---\n\n".join(context_chunks)
        
        # 2. Build a prompt with the chunks as context
        prompt = (
            "You are a helpful and precise assistant. Your task is to answer the user's question "
            "based ONLY on the provided context below. If the context does not contain the answer, "
            "simply state that you do not have enough information to answer.\n\n"
            f"Context Information:\n{context_str}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        
        # 3. Call the LLM to generate an answer
        return self.llm_fn(prompt)
