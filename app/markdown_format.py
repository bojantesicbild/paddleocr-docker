"""Render PP-StructureV3 blocks as HillMetrics-compatible markdown.

Reference format: `/Users/bojantesic/git-tests/HillMetrics.AI/src/DocumentExtraction/temp/*_ocr.md`
Conventions: `> Page N` page markers, GFM tables, <image>[placeholder]</image>,
<logo>...</logo>, <sir>...</sir>. Headings from doc_title / paragraph_title labels.
"""
from __future__ import annotations

from typing import Any

from markdownify import markdownify as _html_to_md

IGNORED_LABELS = {
    "number", "footnote", "header", "header_image",
    "footer", "footer_image", "aside_text", "page_number",
}

IMAGE_LABELS = {"image", "figure", "chart", "flowchart", "header_image", "footer_image"}
FORMULA_LABELS = {"formula", "equation"}

ROW_Y_TOLERANCE = 20.0


def _row_band_key(bbox: list[float]) -> float:
    return round(bbox[1] / ROW_Y_TOLERANCE) * ROW_Y_TOLERANCE


def _sort_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(blocks, key=lambda b: (_row_band_key(b["bbox"]), b["bbox"][0]))


def _table_to_gfm(html: str) -> str:
    gfm = _html_to_md(html, heading_style="ATX").strip()
    return gfm


def _render_block(block: dict[str, Any]) -> str | None:
    label = (block.get("label") or "").lower()
    content = (block.get("content") or "").strip()
    region_id = block.get("region_id")

    if label in IGNORED_LABELS:
        return None

    if label == "doc_title":
        return f"# {content}" if content else None
    if label == "paragraph_title":
        return f"## {content}" if content else None
    if label == "figure_table_chart_title":
        return f"### {content}" if content else None

    if label == "table":
        if content.startswith("<"):
            gfm = _table_to_gfm(content)
            return gfm if gfm else None
        return content or None

    if label in IMAGE_LABELS:
        placeholder = f"[{region_id}]" if region_id else "[region]"
        return f"<image>{placeholder}</image>"

    if label == "seal":
        placeholder = region_id or "region"
        return f"<logo>{placeholder}</logo>"

    if label in FORMULA_LABELS:
        return f"$$\n{content}\n$$" if content else None
    if label == "formula_number":
        return f"${content}$" if content else None

    if label == "reference":
        return "---"

    # text, abstract, content (non-table), reference_content, algorithm, unknown
    return content or None


def render_page(
    blocks: list[dict[str, Any]],
    *,
    page_number: int | None = None,
) -> str:
    parts: list[str] = []
    if page_number is not None:
        parts.append(f"> Page {page_number}")
        parts.append("")
    for block in _sort_blocks(blocks):
        md = _render_block(block)
        if md:
            parts.append(md)
            parts.append("")
    while parts and parts[-1] == "":
        parts.pop()
    return "\n".join(parts)


def render_document(pages: list[list[dict[str, Any]]]) -> str:
    chunks: list[str] = []
    for i, page_blocks in enumerate(pages, start=1):
        chunks.append(render_page(page_blocks, page_number=i))
    return "\n\n".join(chunks)
