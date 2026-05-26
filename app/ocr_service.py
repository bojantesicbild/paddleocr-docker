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
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    s = dict(_load_startup_settings())
    if settings:
        s.update(settings)

    opts = PdfPipelineOptions()
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

    # Device override for local testing on Apple Silicon (MPS doesn't
    # support float64 ops used by some docling stages). On OVH/CUDA leave
    # unset so docling auto-selects gpu:0.
    device = os.environ.get("DOCLING_DEVICE", "").strip()
    if device:
        opts.accelerator_options.device = device

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=opts),
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


def extract(image_bytes: bytes, settings: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = settings or {}
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size

    converter = get_converter()

    # Docling reads from disk. PNG is what /ocr/image and pdf_split feed us.
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

    # Walk pictures in document order, assign stable region_ids, collect crops.
    crops: dict[str, dict[str, Any]] = {}
    picture_to_region: dict[int, str] = {}
    for idx, picture in enumerate(getattr(doc, "pictures", []) or [], start=1):
        rid = f"region_{idx}"
        picture_to_region[id(picture)] = rid

        pil = None
        img_attr = getattr(picture, "image", None)
        if img_attr is not None:
            pil = getattr(img_attr, "pil_image", None)
        if pil is None:
            continue
        crops[rid] = {
            "png": _pil_to_png_bytes(pil),
            "label": _picture_sub_label(picture),
        }

    # Native markdown.
    try:
        markdown = doc.export_to_markdown()
    except Exception:
        markdown = ""

    # Docling emits `![](data:image/png;base64,…)` or `<!-- image -->` for
    # pictures. Rewrite those to our `<image label="…">[region_N]</image>`
    # contract using positional matching: nth picture in the doc → region_N.
    markdown = _rewrite_picture_tags(markdown, doc, picture_to_region, crops)

    return {
        "markdown": markdown,
        "crops": crops,
        "width": w,
        "height": h,
    }


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
