import itertools
from dataclasses import dataclass

from .extract import Block


@dataclass
class ChunkData:
    chunk_index: int
    section: str
    page_start: int
    page_end: int
    text: str


def chunk_blocks(blocks: list[Block], target_words: int = 300, overlap_words: int = 40) -> list[ChunkData]:
    """Section-aware sliding-window chunking. Never crosses a section boundary.

    ~300 words is roughly 400 tokens. Page range comes from the words themselves,
    so citations always carry real page numbers (never LLM-generated ones).
    """
    chunks: list[ChunkData] = []
    step = target_words - overlap_words
    for section, group in itertools.groupby(blocks, key=lambda b: b.section):
        words: list[tuple[str, int]] = []
        for b in group:
            words.extend((w, b.page) for w in b.text.split())
        i = 0
        while i < len(words):
            window = words[i:i + target_words]
            if i > 0 and len(window) <= overlap_words:
                break  # fully covered by the previous chunk's overlap
            if len(window) >= 15:
                chunks.append(ChunkData(
                    chunk_index=len(chunks),
                    section=section,
                    page_start=window[0][1],
                    page_end=window[-1][1],
                    text=" ".join(w for w, _ in window),
                ))
            i += step
    return chunks