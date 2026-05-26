"""Docling-based extraction. Same `extract()` contract as the paddle backend:

    extract(image_bytes, settings) -> {
        markdown: str,
        crops:   {region_id: {"png": bytes, "label": str}},
        width:   int,
        height:  int,
    }

Docling uses smaller, specialized models per task (layout / table-structure /
OCR / picture-classification) and runs on torch. On a V100 (CC 7.0) it is
~10-20× faster than PaddleOCR-VL because there's no autoregressive VLM step.
The trade-off: text *inside* charts isn't transcribed by Docling itself —
only the chart region is labeled. Downstream pipelines can route the cropped
chart to a bigger VL model.
"""
from __future__ import annotations

import io
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from PIL import Image

_converter = None
_converter_lock = threading.Lock()

_SETTINGS_PATH = Path(__file__).resolve().parent.parent / "ocr_settings.json"

# Map Docling DocItemLabel values → the lowercase labels we emit in
# <image label="..."> tags. Keeps the markdown contract stable across
# backend swaps.
_DOC_ITEM_LABEL_MAP = {
    "chart": "chart",
    "picture": "image",
    "table": "table",
    "formula": "formula",
    "page_header": "header",
    "page_footer": "footer",
    "footnote": "footnote",
}


def _load_startup_settings() -> dict[str, Any]:
    try:
        return json.loads(_SETTINGS_PATH.read_text())
    except FileNotFoundError:
        return {}


def _build_converter(settings: dict[str, Any] | None = None):
    """Build a DocumentConverter with the given pipeline settings.

    Heavy: loads layout + table + picture classifier models (~600-800 MB)
    via huggingface_hub on first call per setting combination.

    Uses the ThreadedStandardPdfPipeline for PDF input — runs the 5 pipeline
    stages (preprocess / OCR / layout / table / assemble) concurrently with
    bounded queues + GPU batching. Typically +40% on multi-page docs vs
    the default StandardPdfPipeline.
    """
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import EasyOcrOptions, ThreadedPdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.pipeline.threaded_standard_pdf_pipeline import ThreadedStandardPdfPipeline

    s = dict(_load_startup_settings())
    if settings:
        s.update(settings)

    # AcceleratorOptions: device picked from DOCLING_DEVICE env (cpu/cuda/
    # auto/mps), num_threads tuned via DOCLING_THREADS. Defaults to AUTO
    # which probes CUDA → MPS → XPU → CPU.
    device_env = os.environ.get("DOCLING_DEVICE", "").strip().lower()
    device_map = {
        "cpu": AcceleratorDevice.CPU,
        "cuda": AcceleratorDevice.CUDA,
        "mps": AcceleratorDevice.MPS,
        "xpu": AcceleratorDevice.XPU,
        "auto": AcceleratorDevice.AUTO,
        "": AcceleratorDevice.AUTO,
    }
    acc = AcceleratorOptions(
        device=device_map.get(device_env, AcceleratorDevice.AUTO),
        num_threads=int(os.environ.get("DOCLING_THREADS", "4")),
    )

    opts = ThreadedPdfPipelineOptions(
        accelerator_options=acc,
        # Layout is the most expensive stage after VLM/OCR; batching it
        # across pages is the biggest GPU-utilization win. Default 4; on
        # V100S/A100 with ≥20 GB VRAM, 64 is safe.
        layout_batch_size=int(os.environ.get("DOCLING_LAYOUT_BATCH", "64")),
        ocr_batch_size=int(os.environ.get("DOCLING_OCR_BATCH", "8")),
        table_batch_size=int(os.environ.get("DOCLING_TABLE_BATCH", "4")),
        # Bound worst-case per-document latency. Returns PARTIAL_SUCCESS
        # instead of hanging.
        document_timeout=float(os.environ.get("DOCLING_DOC_TIMEOUT", "120")),
    )
    opts.do_ocr = bool(s.get("do_ocr", True))
    opts.do_table_structure = bool(s.get("do_table_structure", True))
    # Picture-level enrichment is what gives us BAR_CHART / LINE_CHART /
    # LOGO / SIGNATURE / etc. on top of the coarse DocItemLabel.
    opts.do_picture_classification = bool(s.get("do_picture_classification", True))
    # Optional VLM caption of each picture; expensive — off by default.
    opts.do_picture_description = bool(s.get("do_picture_description", False))
    opts.generate_picture_images = True
    opts.images_scale = float(s.get("images_scale", 2.0))

    # Explicitly use EasyOCR. Docling's auto-selection picks RapidOCR for
    # non-English OCR_LANGUAGE, and RapidOCR writes its model cache into
    # its own pip site-packages dir — not writable on OVH AI Deploy's
    # non-root runtime (UID 42420). EasyOCR caches to ~/.EasyOCR which
    # resolves to /tmp/.EasyOCR via HOME=/tmp.
    ocr_langs = s.get("ocr_languages") or ["en", "fr"]
    opts.ocr_options = EasyOcrOptions(lang=ocr_langs)

    return DocumentConverter(
        format_options={
            # Threaded pipeline only for PDFs (where multi-page parallelism
            # matters). Single-image input uses the default pipeline.
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=ThreadedStandardPdfPipeline,
                pipeline_options=opts,
            ),
            InputFormat.IMAGE: PdfFormatOption(pipeline_options=opts),
        }
    )


def get_converter():
    """Singleton DocumentConverter built from startup settings only."""
    global _converter
    if _converter is None:
        with _converter_lock:
            if _converter is None:
                _converter = _build_converter(None)
    return _converter


def _pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    if img.mode != "RGB" and img.mode != "RGBA":
        img = img.convert("RGB")
    img.save(buf, format="PNG")
    return buf.getvalue()


def _picture_sub_label(picture) -> str:
    """Pull the most specific label off a Docling PictureItem:
    prefer the picture-classifier sub-label (BAR_CHART, LOGO, …) over the
    coarse DocItemLabel (CHART, PICTURE)."""
    for ann in getattr(picture, "annotations", []) or []:
        classes = getattr(ann, "predicted_classes", None)
        if classes:
            top = max(classes, key=lambda c: getattr(c, "confidence", 0.0))
            name = getattr(top, "class_name", None)
            if name:
                return name.lower()
    coarse = getattr(picture, "label", None)
    coarse_str = str(coarse).split(".")[-1].lower() if coarse else "image"
    return _DOC_ITEM_LABEL_MAP.get(coarse_str, coarse_str)


def _page_picture_crops(pictures, doc) -> tuple[dict[str, dict[str, Any]], dict[int, str]]:
    """Build per-page crops dict + picture_id→region_id map for placeholder
    rewriting. Region IDs are local to this page (region_1, region_2, …)."""
    crops: dict[str, dict[str, Any]] = {}
    picture_to_region: dict[int, str] = {}
    for idx, picture in enumerate(pictures, start=1):
        rid = f"region_{idx}"
        picture_to_region[id(picture)] = rid
        img_attr = getattr(picture, "image", None)
        pil = getattr(img_attr, "pil_image", None) if img_attr is not None else None
        if pil is None:
            continue
        crops[rid] = {
            "png": _pil_to_png_bytes(pil),
            "label": _picture_sub_label(picture),
        }
    return crops, picture_to_region


def _doc_to_page_result(doc, page_no: int | None = None) -> dict[str, Any]:
    """Render a single page (or the whole doc if page_no is None) into our
    standard {markdown, crops, width, height} shape."""
    # Pictures filtered to this page via prov[0].page_no.
    all_pics = getattr(doc, "pictures", []) or []
    if page_no is not None:
        pics_on_page = [
            p for p in all_pics
            if getattr(p, "prov", None) and getattr(p.prov[0], "page_no", None) == page_no
        ]
    else:
        pics_on_page = list(all_pics)

    crops, picture_to_region = _page_picture_crops(pics_on_page, doc)

    try:
        markdown = doc.export_to_markdown(page_no=page_no) if page_no is not None \
            else doc.export_to_markdown()
    except Exception:
        markdown = ""

    markdown = _rewrite_picture_tags(markdown, doc, picture_to_region, crops)

    # Page dimensions if available.
    w = h = 0
    pages = getattr(doc, "pages", None)
    if pages and page_no in pages:
        size = getattr(pages[page_no], "size", None)
        if size is not None:
            w = int(getattr(size, "width", 0) or 0)
            h = int(getattr(size, "height", 0) or 0)

    return {"markdown": markdown, "crops": crops, "width": w, "height": h}


def extract(image_bytes: bytes, settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """Single-image (or single-page rasterized) extraction. Returns the
    same {markdown, crops, width, height} shape that jobs.py expects."""
    settings = settings or {}
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size

    converter = get_converter()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(image_bytes)
        path = f.name
    try:
        result = converter.convert(path)
        doc = result.document
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    out = _doc_to_page_result(doc, page_no=None)
    # For single images we trust the input PIL dimensions over docling's
    # internal page size (which can be in points, not pixels).
    out["width"] = w
    out["height"] = h
    return out


def extract_pdf(pdf_bytes: bytes, settings: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Multi-page PDF extraction. Hands the PDF directly to docling so the
    text layer (editable PDFs) is used without per-page OCR.

    Returns a list of per-page results, each with the same shape as
    extract(): {markdown, crops, width, height}. Caller (jobs.run_ocr_pdf)
    is responsible for prefixing region_ids per page.
    """
    settings = settings or {}
    converter = get_converter()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(pdf_bytes)
        path = f.name
    try:
        result = converter.convert(path)
        doc = result.document
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    # doc.num_pages is a method (not an attribute) — call it. Fall back to
    # the keys of doc.pages if anything goes sideways.
    try:
        n_pages = int(doc.num_pages())
    except Exception:
        n_pages = len(getattr(doc, "pages", {}) or {}) or 1
    return [_doc_to_page_result(doc, page_no=p) for p in range(1, n_pages + 1)]


def _rewrite_picture_tags(
    markdown: str,
    doc,
    picture_to_region: dict[int, str],
    crops: dict[str, dict[str, Any]],
) -> str:
    """Replace Docling's default picture placeholders in the markdown with
    HillMetrics-style `<image label="…">[region_N]</image>` tags.

    Docling currently writes either `![](data:…)` or `<!-- image -->` /
    a literal placeholder. We replace them positionally with `region_N`
    in the same order the pictures appear in the document.
    """
    import re

    # Build label lookup by region order.
    region_labels: list[tuple[str, str]] = [
        (rid, crops[rid]["label"]) for rid in sorted(crops, key=lambda r: int(r.split("_")[1]))
    ]
    if not region_labels:
        return markdown

    placeholder_re = re.compile(
        r"(!\[[^\]]*\]\([^)]+\)|<!--\s*image\s*-->|<!--\s*figure\s*-->)",
        flags=re.IGNORECASE,
    )

    counter = {"i": 0}

    def _sub(m: re.Match) -> str:
        i = counter["i"]
        counter["i"] += 1
        if i >= len(region_labels):
            return m.group(0)
        rid, label = region_labels[i]
        return f'<image label="{label}">[{rid}]</image>'

    return placeholder_re.sub(_sub, markdown).strip()
