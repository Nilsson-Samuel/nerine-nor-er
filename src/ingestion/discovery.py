"""File discovery — recursively find PDF and DOCX files under a case root."""

from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def discover_documents(case_root: Path) -> list[Path]:
    """Discover PDF and DOCX files recursively under case_root.

    Args:
        case_root: Root directory of the case folder.

    Returns:
        Sorted list of absolute file paths with supported extensions.
    """
    return sorted(
        p for p in case_root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
