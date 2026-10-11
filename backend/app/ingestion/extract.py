import collections
import re
from dataclasses import dataclass

import fitz  # PyMuPDF

SECTION_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?\s+|[IVX]+\.\s+)?"
    r"(abstract|introduction|related work|background|methods?|methodology|"
    r"materials and methods|experiments?|experimental setup|results?|"
    r"results and discussion|discussion|conclusions?|limitations|"
    r"references|bibliography|acknowledg(?:e)?ments?)\s*$",
    re.I,
)
STOP_SECTIONS = {"references", "bibliography"}  # reference lists pollute retrieval


@dataclass
class Block:
    page: int
    section: str
    text: str


def _norm(text: str) -> str:
    return re.sub(r"\d+", "#", text.strip().lower())


def _clean_heading(text: str) -> str:
    return re.sub(r"^(?:\d+(?:\.\d+)*\.?\s+|[IVX]+\.\s+)", "", text.strip()).strip()


def extract_blocks(data: bytes) -> tuple[str | None, list[Block], bool]:
    """Return (title, blocks, low_text_flag). Every block keeps its page number."""
    doc = fitz.open(stream=data, filetype="pdf")
    size_counter: collections.Counter = collections.Counter()
    pages: list[list[dict]] = []

    for page in doc:
        h = page.rect.height
        items = []
        for b in page.get_text("dict")["blocks"]:
            if b.get("type") != 0:
                continue
            lines, sizes, bold_chars, total_chars = [], [], 0, 0
            for line in b["lines"]:
                txt = "".join(s["text"] for s in line["spans"]).strip()
                if txt:
                    lines.append(txt)
                for s in line["spans"]:
                    n = len(s["text"].strip())
                    if n:
                        size_counter[round(s["size"])] += n
                        sizes.append(s["size"])
                        total_chars += n
                        if s["flags"] & 16:
                            bold_chars += n
            if not lines:
                continue
            y0, y1 = b["bbox"][1], b["bbox"][3]
            items.append({
                "text": " ".join(lines),
                "nlines": len(lines),
                "size": max(sizes),
                "bold": total_chars > 0 and bold_chars / total_chars > 0.8,
                "edge": y1 < h * 0.08 or y0 > h * 0.92,
            })
        pages.append(items)

    body_size = size_counter.most_common(1)[0][0] if size_counter else 10

    # Headers/footers: text (digits normalised) in the top/bottom margin on >=50% of pages.
    edge_counter: collections.Counter = collections.Counter()
    for items in pages:
        edge_counter.update({_norm(i["text"]) for i in items if i["edge"]})
    repeated = {k for k, v in edge_counter.items() if v >= max(2, len(pages) * 0.5)}

    # Title: largest-font block on page 1 (heuristic; verify on your corpus).
    title = None
    if pages and pages[0]:
        cand = max(pages[0], key=lambda i: i["size"])
        if len(cand["text"]) < 250:
            title = cand["text"]

    blocks: list[Block] = []
    section = "Front matter"
    total_words = 0
    for pno, items in enumerate(pages, start=1):
        for it in items:
            text = it["text"]
            if it["edge"] and _norm(text) in repeated:
                continue
            short = it["nlines"] <= 2 and len(text) < 100 and not text.endswith(".")
            is_heading = bool(SECTION_RE.match(text)) or (
                short and (it["size"] >= body_size * 1.15 or (it["bold"] and len(text.split()) <= 8))
            )
            if is_heading:
                name = _clean_heading(text)
                if name.lower() in STOP_SECTIONS:
                    return title, blocks, total_words < 50 * len(pages)
                section = name[:200]
                continue
            blocks.append(Block(page=pno, section=section, text=text))
            total_words += len(text.split())

    # Very little text per page usually means a scanned PDF (no OCR in the MVP).
    return title, blocks, total_words < 50 * len(pages)