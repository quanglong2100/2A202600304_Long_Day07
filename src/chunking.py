from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return[]
        
        # Split but keep the delimiters by using a capturing group
        parts = re.split(r'([.!?] |\.\n)', text)
        sentences =[]
        current_sentence = ""
        
        for part in parts:
            current_sentence += part
            # Check if this part was a delimiter
            if part in[". ", "! ", "? ", ".\n"]:
                stripped = current_sentence.strip()
                if stripped:
                    sentences.append(stripped)
                current_sentence = ""
                
        # Append the last chunk if it didn't end with a delimiter
        if current_sentence.strip():
            sentences.append(current_sentence.strip())

        # Group sentences into chunks
        chunks =[]
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunk = " ".join(sentences[i : i + self.max_sentences_per_chunk])
            chunks.append(chunk)
            
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if not current_text:
            return[]
        if len(current_text) <= self.chunk_size:
            return [current_text]

        # Find the highest priority separator that actually exists in the text
        separator = ""
        new_separators =[]
        for i, sep in enumerate(remaining_separators):
            if sep == "":
                separator = sep
                break
            if sep in current_text:
                separator = sep
                new_separators = remaining_separators[i + 1:]
                break

        # Fallback if the separator is an empty string (meaning character-level split)
        if separator == "":
            return[current_text[i : i + self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]

        # Split the text
        splits = current_text.split(separator)
        chunks =[]
        current_chunk = ""

        for part in splits:
            if not current_chunk:
                candidate = part
            else:
                candidate = current_chunk + separator + part

            # If adding this part stays within the chunk size, merge it
            if len(candidate) <= self.chunk_size:
                current_chunk = candidate
            else:
                # Store the current chunk since adding `part` exceeds the limit
                if current_chunk:
                    chunks.append(current_chunk)
                    
                # If `part` itself is too large, recurse!
                if len(part) > self.chunk_size:
                    if new_separators:
                        chunks.extend(self._split(part, new_separators))
                    else:
                        chunks.extend([part[i : i + self.chunk_size] for i in range(0, len(part), self.chunk_size)])
                    current_chunk = ""
                else:
                    current_chunk = part

        if current_chunk:
            chunks.append(current_chunk)

        return chunks



def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    if not vec_a or not vec_b:
        return 0.0
        
    dot_product = _dot(vec_a, vec_b)
    mag_a = math.sqrt(_dot(vec_a, vec_a))
    mag_b = math.sqrt(_dot(vec_b, vec_b))
    
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
        
    return dot_product / (mag_a * mag_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        fixed_chunker = FixedSizeChunker(chunk_size=chunk_size)
        sentence_chunker = SentenceChunker(max_sentences_per_chunk=3)
        recursive_chunker = RecursiveChunker(chunk_size=chunk_size)

        fixed_res = fixed_chunker.chunk(text)
        sentence_res = sentence_chunker.chunk(text)
        recursive_res = recursive_chunker.chunk(text)

        return {
            "fixed_size": {
                "count": len(fixed_res),
                "avg_length": sum(len(c) for c in fixed_res) / len(fixed_res) if fixed_res else 0,
                "chunks": fixed_res
            },
            "by_sentences": {  # <--- ĐỔI 'sentence' THÀNH 'by_sentences' Ở ĐÂY
                "count": len(sentence_res),
                "avg_length": sum(len(c) for c in sentence_res) / len(sentence_res) if sentence_res else 0,
                "chunks": sentence_res
            },
            "recursive": {
                "count": len(recursive_res),
                "avg_length": sum(len(c) for c in recursive_res) / len(recursive_res) if recursive_res else 0,
                "chunks": recursive_res
            }
        }