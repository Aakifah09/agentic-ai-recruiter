"""Document ingestion: PDF, DOCX and plain text -> clean text."""

from __future__ import annotations

import io
import re
import unicodedata
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
MAX_BYTES = 10 * 1024 * 1024


class DocumentError(ValueError):
    pass


def extract_text(data: bytes, filename: str) -> str:
    """Extract text from raw file bytes, dispatching on the file extension."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise DocumentError(f"Unsupported file type '{suffix}'. Use PDF, DOCX or TXT.")
    if len(data) > MAX_BYTES:
        raise DocumentError(f"{filename} is larger than 10 MB.")
    if not data:
        raise DocumentError(f"{filename} is empty.")

    if suffix == ".pdf":
        text = _pdf_text(data, filename)
    elif suffix == ".docx":
        text = _docx_text(data, filename)
    else:
        text = _decode(data)

    text = clean_text(text)
    if len(text) < 40:
        raise DocumentError(
            f"Could not extract readable text from {filename}. Scanned/image-only documents need OCR first."
        )
    return text


def extract_file(path: str | Path) -> str:
    path = Path(path)
    return extract_text(path.read_bytes(), path.name)


def _pdf_text(data: bytes, filename: str) -> str:
    import fitz  # PyMuPDF

    try:
        with fitz.open(stream=data, filetype="pdf") as doc:
            # sort=True keeps multi-column layouts in reading order.
            return "\n".join(page.get_text("text", sort=True) for page in doc)
    except Exception as exc:
        raise DocumentError(f"Could not read PDF {filename}: {exc}") from exc


def _docx_text(data: bytes, filename: str) -> str:
    import docx

    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:
        raise DocumentError(f"Could not read DOCX {filename}: {exc}") from exc
    lines = [p.text for p in document.paragraphs]
    for table in document.tables:  # many resume templates put skills in tables
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                lines.append(" | ".join(dict.fromkeys(cells)))
    return "\n".join(lines)


def _decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


_BULLETS = re.compile(r"^ *[•●▪◦‣∙·\-\*–—]+ *", re.MULTILINE)  # tabs are already spaces


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    text = _BULLETS.sub("- ", text)
    text = re.sub(r"[ \u00a0]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()
