"""Conservative, auditable identity normalization for song matching."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

import pandas as pd
from unidecode import unidecode

_SPACE = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")
_TITLE_VERSION = re.compile(
    r"\s*(?:\(|\[|-)?\s*(?:remaster(?:ed)?(?:\s+\d{4})?|radio\s+edit|"
    r"live|single\s+version|album\s+version)\s*(?:\)|\])?\s*$",
    flags=re.IGNORECASE,
)


def normalize_key(value: Any, *, transliterate: bool = True) -> str:
    """Return a stable comparison key while leaving source strings untouched."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    if transliterate:
        text = unidecode(text)
    text = _SPACE.sub(" ", text)
    text = _PUNCT.sub("", text)
    return _SPACE.sub(" ", text).strip()


def base_title(value: Any, *, strip_versions: bool = False) -> str:
    """Build a title key; suffix removal is opt-in and intentionally narrow."""
    text = "" if value is None else str(value)
    if strip_versions:
        text = _TITLE_VERSION.sub("", text)
    return normalize_key(text)


def canonicalize_frame(
    frame: pd.DataFrame,
    *,
    artist_col: str = "artist_name",
    title_col: str = "title",
    strip_versions: bool = False,
) -> pd.DataFrame:
    """Add original-preserving matching keys to a copy of *frame*."""
    required = {artist_col, title_col}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Cannot canonicalize missing columns: {sorted(missing)}")
    out = frame.copy()
    out["artist_key"] = out[artist_col].map(normalize_key)
    out["title_key"] = out[title_col].map(
        lambda value: base_title(value, strip_versions=strip_versions)
    )
    out["canonical_key"] = out["artist_key"] + " || " + out["title_key"]
    return out
