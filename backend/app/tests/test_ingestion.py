import os

os.environ.setdefault("DATABASE_URL", "postgresql://x:x@localhost/x")

import pymupdf as fitz
import pytest

from app.ingestion.chunk import chunk_blocks
from app.ingestion.extract import Block, extract_blocks
from app.ingestion.validate import InvalidPDF, validate_pdf


def make_pdf() -> bytes:
    doc = fitz.open()
    for pno in range(1, 4):
        page = doc.new_page()
        page.insert_text((72, 30), "Journal of Tests", fontsize=9)           # repeated header
        page.insert_text((72, 820), f"Page {pno}", fontsize=9)                # footer
        if pno == 1:
            page.insert_text((72, 100), "Introduction", fontsize=16)
        if pno == 3:
            page.insert_text((72, 100), "References", fontsize=16)
        y = 130
        for _ in range(25):
            page.insert_text((72, y), "Random forests outperformed baselines on the drought dataset. " * 1, fontsize=10)
            y += 14
    return doc.tobytes()


def test_validate_rejects_non_pdf():
    with pytest.raises(InvalidPDF):
        validate_pdf(b"not a pdf")


def test_extract_strips_headers_and_stops_at_references():
    title, blocks, low = extract_blocks(make_pdf())
    assert not low
    assert all("Journal of Tests" not in b.text for b in blocks)
    assert all(b.page < 3 or b.section != "References" for b in blocks)
    assert any(b.section == "Introduction" for b in blocks)


def test_chunks_respect_sections_and_pages():
    blocks = [Block(1, "Intro", "word " * 700), Block(2, "Intro", "word " * 100),
              Block(2, "Methods", "alpha " * 100)]
    chunks = chunk_blocks(blocks, target_words=300, overlap_words=40)
    assert {c.section for c in chunks} == {"Intro", "Methods"}
    assert all(len(c.text.split()) <= 300 for c in chunks)
    assert chunks[0].page_start == 1 and chunks[-1].section == "Methods"
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))