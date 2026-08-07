"""Document loading and chunking.

Chunk size is tuned for *spoken* answers rather than written ones. A 1500-token
chunk gives a chat model lovely context and gives a voice agent a rambling
twenty-second monologue. ~700 characters is about one spoken paragraph, which is
the largest unit a caller can absorb by ear.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vaani.core.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class Chunk:
    text: str
    source: str
    ordinal: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


_PARAGRAPH = re.compile(r"\n\s*\n")
# Sentence terminators including the Devanagari danda and Arabic full stop.
_SENTENCE = re.compile(r"(?<=[.!?।۔])\s+")


def chunk_text(
    text: str,
    source: str,
    *,
    target_chars: int = 700,
    overlap_chars: int = 120,
    metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Split on paragraph boundaries, then sentences, never mid-word.

    Overlap carries the tail of each chunk into the next so a fact split across
    a boundary is still retrievable from at least one side.
    """
    text = _normalise(text)
    if not text:
        return []

    units: list[str] = []
    for para in _PARAGRAPH.split(text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= target_chars:
            units.append(para)
            continue
        # Oversized paragraph: fall back to sentences.
        buf = ""
        for sentence in _SENTENCE.split(para):
            if len(buf) + len(sentence) + 1 > target_chars and buf:
                units.append(buf.strip())
                buf = sentence
            else:
                buf = f"{buf} {sentence}".strip()
        if buf.strip():
            units.append(buf.strip())

    chunks: list[Chunk] = []
    buf = ""
    for unit in units:
        if len(buf) + len(unit) + 1 > target_chars and buf:
            chunks.append(buf.strip())
            buf = (buf[-overlap_chars:] + " " + unit) if overlap_chars else unit
        else:
            buf = f"{buf}\n\n{unit}".strip() if buf else unit
    if buf.strip():
        chunks.append(buf.strip())

    kept = [c for c in chunks if _is_substantive(c, len(chunks))]
    return [
        Chunk(text=c, source=source, ordinal=i, metadata=dict(metadata or {}))
        for i, c in enumerate(kept)
    ]


def _is_substantive(chunk: str, total_chunks: int) -> bool:
    """Filter out page furniture without discarding genuinely short facts.

    A bare length threshold is wrong: "Tariff slab one is five rupees." is 31
    characters and is exactly the kind of fact a caller rings up to ask about,
    while "CHAPTER IV" is 10 characters of noise. So a short fragment is dropped
    only when it sits among other chunks — where it is almost certainly a heading
    — and a document that is short in its entirety is always kept.
    """
    text = chunk.strip()
    if len(text) < 12:
        return False
    if not re.search(r"[A-Za-zऀ-෿؀-ۿ]", text):
        return False  # numbers, dates or symbols alone
    if total_chunks == 1:
        return True
    # Among siblings: a heading has no sentence structure. Require either
    # reasonable length or an actual sentence.
    return len(text) >= 40 or bool(re.search(r"[.!?।۔]", text))


def _normalise(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip page-number-only lines left by PDF extraction.
    text = re.sub(r"^\s*\d{1,4}\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix in (".txt", ".md", ".csv", ".json", ".html", ".htm"):
        raw = path.read_text(encoding="utf-8", errors="replace")
        return _strip_html(raw) if suffix in (".html", ".htm") else raw
    if suffix == ".docx":
        return _load_docx(path)
    raise ValueError(f"Unsupported document type: {suffix}")


def _load_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF ingestion needs `pip install vaani[rag]`") from exc

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(pages)
    if len(text.strip()) < 50 * len(pages) / 2:
        # Almost no extractable text: this is a scan. OCR is a separate,
        # heavier path — flag it rather than silently indexing empty pages.
        log.warning(
            "pdf appears to be scanned; OCR required",
            extra={"path": str(path), "pages": len(pages)},
        )
    return text


def _load_docx(path: Path) -> str:
    try:
        import docx  # python-docx
    except ImportError as exc:
        raise RuntimeError("DOCX ingestion needs `pip install python-docx`") from exc
    document = docx.Document(str(path))
    return "\n\n".join(p.text for p in document.paragraphs if p.text.strip())


def _strip_html(raw: str) -> str:
    raw = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.S | re.I)
    raw = re.sub(r"<[^>]+>", " ", raw)
    return re.sub(r"&nbsp;?", " ", raw)


def chunk_files(paths: Iterable[Path], **kwargs: Any) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in paths:
        try:
            chunks.extend(chunk_text(load_file(path), source=path.name, **kwargs))
        except Exception:
            log.exception("failed to ingest", extra={"path": str(path)})
    return chunks
