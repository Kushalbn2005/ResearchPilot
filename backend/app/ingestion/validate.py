import pymupdf as fitz  # PyMuPDF

from ..config import settings


class InvalidPDF(ValueError):
    pass


def validate_pdf(data: bytes) -> int:
    """Return page count, or raise InvalidPDF. Never trusts the filename or content-type."""
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise InvalidPDF(f"File exceeds {settings.max_upload_mb} MB")
    if not data.startswith(b"%PDF"):
        raise InvalidPDF("Not a PDF file")
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception:
        raise InvalidPDF("PDF is corrupt or unreadable")
    if doc.needs_pass:
        raise InvalidPDF("Encrypted PDFs are not supported")
    if doc.page_count == 0:
        raise InvalidPDF("PDF has no pages")
    if doc.page_count > settings.max_pdf_pages:
        raise InvalidPDF(f"PDF has more than {settings.max_pdf_pages} pages")
    return doc.page_count