"""PP-StructureV3 extraction. Ported from ocr_worker.py to return in-memory crops."""
from __future__ import annotations

import io
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from PIL import Image

_pipe = None
_pipe_lock = threading.Lock()

VISUAL_LABELS = {
    "figure", "image", "chart", "equation",
    "header_image", "footer_image", "flowchart", "table",
}


def get_pipeline():
    global _pipe
    if _pipe is None:
        with _pipe_lock:
            if _pipe is None:
                from paddleocr import PPStructureV3

                lang = os.environ.get("OCR_LANGUAGE", "fr")
                _pipe = PPStructureV3(lang=lang)
    return _pipe


def _bbox_overlap_ratio(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = max((a[2] - a[0]) * (a[3] - a[1]), 1)
    return inter / area_a


def _crop_png_bytes(img: Image.Image, bbox: list[float]) -> bytes | None:
    w, h = img.size
    x1, y1, x2, y2 = (int(round(c)) for c in bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    buf = io.BytesIO()
    img.crop((x1, y1, x2, y2)).save(buf, format="PNG")
    return buf.getvalue()


def extract(image_bytes: bytes, settings: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = settings or {}
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size

    pipe = get_pipeline()
    predict_kwargs = {
        "use_doc_orientation_classify": settings.get("use_doc_orientation_classify", False),
        "use_doc_unwarping": settings.get("use_doc_unwarping", False),
        "use_seal_recognition": settings.get("use_seal_recognition", False),
        "use_table_recognition": settings.get("use_table_recognition", True),
        "use_formula_recognition": settings.get("use_formula_recognition", False),
        "use_chart_recognition": settings.get("chart_recognition", False),
        "layout_threshold": settings.get("layout_threshold", 0.3),
        "use_e2e_wired_table_rec_model": settings.get("use_e2e_wired_table_rec_model", False),
        "use_e2e_wireless_table_rec_model": settings.get("use_e2e_wireless_table_rec_model", False),
        "use_wired_table_cells_trans_to_html": settings.get("use_wired_table_cells_trans_to_html", True),
        "use_wireless_table_cells_trans_to_html": settings.get("use_wireless_table_cells_trans_to_html", True),
        "use_ocr_results_with_table_cells": settings.get("use_ocr_results_with_table_cells", True),
    }

    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        img_path = tmp_dir / "page.png"
        img.save(img_path, format="PNG")
        results = pipe.predict(str(img_path), **predict_kwargs)
        if not results:
            return {"blocks": [], "text_lines": [], "width": w, "height": h, "crops": {}}

        r = results[0]
        json_dir = tmp_dir / "layout_json"
        json_dir.mkdir(exist_ok=True)
        r.save_to_json(str(json_dir))
        saved_json = json_dir / f"{img_path.stem}_res.json"
        data = json.loads(saved_json.read_text())

    blocks: list[dict[str, Any]] = []
    parsing_list = data.get("parsing_res_list", [])

    crops: dict[str, bytes] = {}

    for i, block in enumerate(parsing_list):
        bbox = block.get("block_bbox", [0, 0, 0, 0])
        label = block.get("block_label", "unknown")
        entry = {
            "label": label,
            "bbox": [round(c, 1) for c in bbox],
            "content": block.get("block_content", ""),
            "block_idx": i,
        }
        if label.lower() in VISUAL_LABELS:
            png = _crop_png_bytes(img, bbox)
            if png is not None:
                region_id = f"region_{i}"
                crops[region_id] = png
                entry["region_id"] = region_id
        blocks.append(entry)

    # Deduplicate overlapping blocks (drop smaller block overlapping a larger one >60%)
    keep = [True] * len(blocks)
    for i in range(len(blocks)):
        if not keep[i]:
            continue
        for j in range(len(blocks)):
            if i == j or not keep[j]:
                continue
            area_i = (blocks[i]["bbox"][2] - blocks[i]["bbox"][0]) * (blocks[i]["bbox"][3] - blocks[i]["bbox"][1])
            area_j = (blocks[j]["bbox"][2] - blocks[j]["bbox"][0]) * (blocks[j]["bbox"][3] - blocks[j]["bbox"][1])
            if area_i <= area_j and _bbox_overlap_ratio(blocks[i]["bbox"], blocks[j]["bbox"]) > 0.6:
                keep[i] = False
                break
    blocks = [b for b, k in zip(blocks, keep) if k]
    for i, b in enumerate(blocks):
        b["block_idx"] = i

    # Extract text lines and tie them to blocks
    text_lines: list[dict[str, Any]] = []
    ocr_res = data.get("overall_ocr_res", {})
    rec_texts = ocr_res.get("rec_texts", [])
    rec_polys = ocr_res.get("rec_polys", [])
    rec_scores = ocr_res.get("rec_scores", [])

    for i in range(len(rec_texts)):
        poly = [[int(p[0]), int(p[1])] for p in rec_polys[i]]
        line_cx = (poly[0][0] + poly[2][0]) / 2
        line_cy = (poly[0][1] + poly[2][1]) / 2
        block_idx = -1
        for bi, blk in enumerate(blocks):
            bx1, by1, bx2, by2 = blk["bbox"]
            if bx1 <= line_cx <= bx2 and by1 <= line_cy <= by2:
                block_idx = bi
                break
        text_lines.append({
            "text": rec_texts[i],
            "bbox": poly,
            "score": round(float(rec_scores[i]), 4),
            "block_idx": block_idx,
        })

    lines_by_block: dict[int, list[dict[str, Any]]] = {}
    for tl in text_lines:
        bi = tl["block_idx"]
        if bi >= 0:
            lines_by_block.setdefault(bi, []).append(tl)

    # Rebuild block content from text lines (skip HTML table / structured chart content)
    for bi, block_lines in lines_by_block.items():
        if bi >= len(blocks):
            continue
        existing = blocks[bi].get("content", "") or ""
        if existing.startswith("<html") or existing.startswith("<table"):
            continue
        if blocks[bi]["label"] == "chart" and "|" in existing:
            continue
        blocks[bi]["content"] = " ".join(tl["text"] for tl in block_lines)

    # Detect tabular structure in "content" blocks: 4+ lines forming rows with 2+ cells
    for bi, blk in enumerate(blocks):
        if blk["label"] != "content":
            continue
        block_lines = lines_by_block.get(bi, [])
        if len(block_lines) < 4:
            continue
        sorted_lines = sorted(block_lines, key=lambda tl: (tl["bbox"][0][1] + tl["bbox"][2][1]) / 2)
        rows: list[list[dict[str, Any]]] = []
        current_row = [sorted_lines[0]]
        for tl in sorted_lines[1:]:
            prev_cy = sum((l["bbox"][0][1] + l["bbox"][2][1]) / 2 for l in current_row) / len(current_row)
            cur_cy = (tl["bbox"][0][1] + tl["bbox"][2][1]) / 2
            if abs(cur_cy - prev_cy) < 15:
                current_row.append(tl)
            else:
                rows.append(current_row)
                current_row = [tl]
        rows.append(current_row)
        multi_cell_rows = sum(1 for r in rows if len(r) >= 2)
        if multi_cell_rows < 3:
            continue
        html_parts = ["<table><tbody>"]
        for row in rows:
            row_sorted = sorted(row, key=lambda tl: tl["bbox"][0][0])
            html_parts.append("<tr>")
            for cell in row_sorted:
                html_parts.append(f"<td>{cell['text']}</td>")
            html_parts.append("</tr>")
        html_parts.append("</tbody></table>")
        blk["content"] = "".join(html_parts)
        blk["label"] = "table"
        if "region_id" not in blk:
            png = _crop_png_bytes(img, blk["bbox"])
            if png is not None:
                region_id = f"region_{bi}"
                crops[region_id] = png
                blk["region_id"] = region_id

    return {
        "blocks": blocks,
        "text_lines": text_lines,
        "width": w,
        "height": h,
        "crops": crops,
    }
