"""
core/source_reader.py
Unified text extraction for source documents referenced by
position.responsibility_source (e.g. offer letters, JDs) during onboarding.

Supports .txt, .md, .pdf, and url. Shared across feature modules -- cv_gen or
jd_collector can reuse this if they ever need to read uploaded PDFs too.
"""

from __future__ import annotations

from pathlib import Path


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


class UnsupportedFileTypeError(ValueError):
    pass


class FileReadError(RuntimeError):
    pass


def read_source(source: str) -> str:
    """統一入口：自動判斷是本地檔案路徑還是 URL，回傳純文字內容。"""
    if _is_url(source):
        return _read_url(source)
    return read_source_file(source)


def read_source_file(file_path: str) -> str:
    """
    Reads and returns the text content of a .txt, .md, or .pdf file.
    Raises UnsupportedFileTypeError for other extensions, and
    FileReadError if the file is missing or fails to parse.
    """
    path = Path(file_path).expanduser().resolve()

    if not path.exists():
        raise FileReadError(f"File not found: {path}")
    if not path.is_file():
        raise FileReadError(f"Not a file: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    try:
        if suffix in (".txt", ".md"):
            return _read_plain_text(path)
        return _read_pdf(path)
    except (UnsupportedFileTypeError, FileReadError):
        raise
    except Exception as exc:  # noqa: BLE001 -- surface as a single, clear error type
        raise FileReadError(f"Failed to read {path}: {exc}") from exc


def _read_plain_text(path: Path) -> str:
    content = path.read_text(encoding="utf-8")
    if not content.strip():
        raise FileReadError(f"File is empty: {path}")
    return content


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader  # imported lazily so txt/md-only usage has no pypdf dependency

    reader = PdfReader(str(path))
    pages_text = [page.extract_text() or "" for page in reader.pages]
    content = "\n".join(pages_text).strip()

    if not content:
        raise FileReadError(
            f"No extractable text found in {path}. "
            "This may be a scanned/image-based PDF, which requires OCR (not yet supported)."
        )
    return content

def _read_url(url: str, timeout: int = 10) -> str:
    import requests
    import trafilatura

    try:
        resp = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (compatible; find-a-job-bot/1.0)"},
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        raise ValueError(f"URL Unreadable {url}: {e}") from e

    text = trafilatura.extract(resp.text, url=url)
    if not text:
        raise ValueError(f"URL {url} Unparsable")
    return text.strip()

def _is_url(source: str) -> bool:
    import re
    
    URL_PATTERN = re.compile(r"^https?://", re.IGNORECASE)

    return bool(URL_PATTERN.match(source.strip()))
