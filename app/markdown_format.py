"""Page/document assembly for PaddleOCR-VL output.

The VL pipeline emits high-quality page markdown directly (tables as GFM,
formulas as LaTeX). This module only adds HillMetrics `> Page N` markers and
(for PDFs) prefixes region_ids per page so they stay unique across pages.
"""
from __future__ import annotations

import re

from bs4 import BeautifulSoup

_REGION_TAG_RE = re.compile(r'<image(\s[^>]*)?>\[(region_\d+)\]</image>')
_HTML_TABLE_RE = re.compile(r"<table\b[^>]*>.*?</table>", re.IGNORECASE | re.DOTALL)


def _cell_text(cell) -> str:
    # Flatten cell contents to a single line: newlines break GFM tables.
    text = cell.get_text(" ", strip=True)
    # Escape pipes so they don't collide with GFM delimiters.
    return text.replace("|", "\\|")


def _table_to_gfm(html: str) -> str:
    """Convert a single <table>...</table> to a GFM pipe table.

    Uses the first <tr> as the header row (promoting <td> to header if the
    source has no <thead>/<th>). Expands colspan by cell duplication; rowspan
    is ignored (cells leak upward — GFM can't express row-spans).
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        return html

    rows = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue
        expanded: list[str] = []
        for c in cells:
            text = _cell_text(c)
            try:
                span = int(c.get("colspan") or 1)
            except ValueError:
                span = 1
            expanded.extend([text] * max(1, span))
        rows.append(expanded)

    if not rows:
        return html

    ncols = max(len(r) for r in rows)
    rows = [r + [""] * (ncols - len(r)) for r in rows]

    header, body = rows[0], rows[1:]
    # If the first row looks empty, treat the second as the header instead.
    if all(not c.strip() for c in header) and body:
        header, body = body[0], body[1:]

    def _line(cells: list[str]) -> str:
        return "| " + " | ".join(c if c else " " for c in cells) + " |"

    lines = [_line(header), "| " + " | ".join(["---"] * ncols) + " |"]
    lines.extend(_line(r) for r in body)
    return "\n".join(lines)


def html_tables_to_gfm(markdown: str) -> str:
    """Replace every <table>...</table> block in `markdown` with a GFM table.

    The VL output embeds tables as HTML without a <thead>; naive conversion
    produces an empty header row. We use the first <tr> as the header.
    VL already separates blocks with \\n\\n, so we do not add surrounding
    blank lines around the converted table — that would stack into 3+ blank
    lines and break spacing.
    """
    def _sub(m: re.Match) -> str:
        gfm = _table_to_gfm(m.group(0))
        return gfm if gfm else m.group(0)

    return _HTML_TABLE_RE.sub(_sub, markdown)


_BLANKS_RE = re.compile(r"\n{3,}")


def collapse_blank_lines(markdown: str) -> str:
    """Collapse runs of 3+ newlines down to two (one blank line between blocks)."""
    return _BLANKS_RE.sub("\n\n", markdown)


def prefix_region_ids(markdown: str, prefix: str) -> tuple[str, dict[str, str]]:
    """Prefix every `region_N` id in the markdown with `prefix`.

    Returns (new_markdown, {old_id: new_id}).
    """
    mapping: dict[str, str] = {}

    def _sub(m: re.Match) -> str:
        attrs = m.group(1) or ""
        old = m.group(2)
        new = prefix + old
        mapping[old] = new
        return f"<image{attrs}>[{new}]</image>"

    return _REGION_TAG_RE.sub(_sub, markdown), mapping


def render_page(markdown: str, *, page_number: int | None = None) -> str:
    body = markdown.strip()
    if page_number is None:
        return body
    if not body:
        return f"> Page {page_number}"
    return f"> Page {page_number}\n\n{body}"
