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

# Cache of DocumentConverters keyed by a hash of the build-affecting
# settings. The first build per key downloads models + warms torch; later
# requests with the same effective settings reuse the cached converter.
# Capped at 3 entries — a 4th distinct setting combination evicts the
# least-recently-used. Bumping higher costs ~600-800 MB of RAM each.
_CONVERTER_CACHE_MAX = 3
_converter_cache: "collections.OrderedDict[str, Any]" = None  # type: ignore[assignment]
_converter_lock = threading.Lock()

_SETTINGS_PATH = Path(__file__).resolve().parent.parent / "ocr_settings.json"

# Settings keys that affect how the DocumentConverter is built. Toggling
# any of these requires a rebuild (loads/unloads pipeline stages, swaps
# OCR model, changes batch sizes, etc.). Anything not listed here is a
# no-op at convert-time because docling reads it during build.
_BUILD_AFFECTING_KEYS = (
    "do_ocr",
    "do_table_structure",
    "do_picture_classification",
    "do_picture_description",
    "do_chart_extraction",
    "generate_table_images",
    "images_scale",
    "ocr_languages",
    "ocr_confidence_threshold",
    "picture_description_repo_id",
    "picture_description_prompt",
    # do_chart_to_table is OURS, not docling's — runs DePlot in a
    # post-pass. Doesn't change the converter, but keying on it keeps
    # the conceptual settings model uniform (a request with VLM+DePlot
    # is distinct from one without).
    "do_chart_to_table",
)

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
    from docling.datamodel.pipeline_options import (
        EasyOcrOptions,
        PdfPipelineOptions,
        PictureDescriptionVlmOptions,
        ThreadedPdfPipelineOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

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
        # CC 7.0 (V100) has no FA2 support — keep explicit so docling never
        # tries to enable it on Ampere-and-up hosts when we share the image.
        cuda_use_flash_attention2=False,
    )

    use_threaded = os.environ.get("DOCLING_PIPELINE", "").strip().lower() == "threaded"
    doc_timeout = float(os.environ.get("DOCLING_DOC_TIMEOUT", "300"))

    if use_threaded:
        from docling.pipeline.threaded_standard_pdf_pipeline import ThreadedStandardPdfPipeline
        opts = ThreadedPdfPipelineOptions(
            accelerator_options=acc,
            layout_batch_size=int(os.environ.get("DOCLING_LAYOUT_BATCH", "4")),
            ocr_batch_size=int(os.environ.get("DOCLING_OCR_BATCH", "8")),
            table_batch_size=int(os.environ.get("DOCLING_TABLE_BATCH", "4")),
            document_timeout=doc_timeout,
        )
        pipeline_cls = ThreadedStandardPdfPipeline
    else:
        # Default = StandardPdfPipeline. The threaded variant has shown
        # CUDA failures on Tesla V100S; opt in only when verified.
        opts = PdfPipelineOptions(accelerator_options=acc, document_timeout=doc_timeout)
        pipeline_cls = None

    opts.do_ocr = bool(s.get("do_ocr", True))
    opts.do_table_structure = bool(s.get("do_table_structure", True))
    # Picture-level enrichment is what gives us BAR_CHART / LINE_CHART /
    # LOGO / SIGNATURE / etc. on top of the coarse DocItemLabel.
    opts.do_picture_classification = bool(s.get("do_picture_classification", True))
    # Optional VLM caption of each picture; expensive — off by default.
    # Configurable repo_id: defaults to SmolVLM-256M (fast, ~600 MB).
    # Swap to "ibm-granite/granite-docling-258M" for doc-tuned VLM or
    # "ibm-granite/granite-vision-3.3-2b" for higher quality at 2B-param cost.
    opts.do_picture_description = bool(s.get("do_picture_description", False))
    if opts.do_picture_description:
        opts.picture_description_options = PictureDescriptionVlmOptions(
            repo_id=s.get("picture_description_repo_id") or "HuggingFaceTB/SmolVLM-256M-Instruct",
            prompt=s.get("picture_description_prompt") or "Describe this image in a few sentences.",
        )
    # Chart extraction (docling 2.72+): VLM-based chart2csv / chart2code on
    # detected chart regions. Heavy — pulls granite-vision-v4. Off by default.
    opts.do_chart_extraction = bool(s.get("do_chart_extraction", False))
    opts.generate_picture_images = True
    # Table crops join the `images` array when enabled — gives downstream a
    # fallback when TableFormer mis-parses a complex layout.
    opts.generate_table_images = bool(s.get("generate_table_images", True))
    opts.images_scale = float(s.get("images_scale", 2.0))

    # Explicitly use EasyOCR. Docling's auto-selection picks RapidOCR for
    # non-English OCR_LANGUAGE, and RapidOCR writes its model cache into
    # its own pip site-packages dir — not writable on OVH AI Deploy's
    # non-root runtime (UID 42420). EasyOCR caches to ~/.EasyOCR which
    # resolves to /tmp/.EasyOCR via HOME=/tmp.
    # Lang order matters: EasyOCR's recognizer biases toward the first lang
    # for ambiguous glyphs (e.g. `ä` vs `a`). Put the dominant doc lang first.
    # confidence_threshold=0.3 (vs default 0.5) recovers low-confidence
    # umlauts/accents that would otherwise drop on DE/FR text.
    ocr_langs = s.get("ocr_languages") or ["de", "fr", "en"]
    opts.ocr_options = EasyOcrOptions(
        lang=ocr_langs,
        confidence_threshold=float(s.get("ocr_confidence_threshold", 0.3)),
    )

    pdf_fmt = PdfFormatOption(pipeline_cls=pipeline_cls, pipeline_options=opts) \
        if pipeline_cls else PdfFormatOption(pipeline_options=opts)
    return DocumentConverter(
        format_options={
            InputFormat.PDF: pdf_fmt,
            InputFormat.IMAGE: PdfFormatOption(pipeline_options=opts),
        }
    )


def _effective_settings(settings: dict[str, Any] | None) -> dict[str, Any]:
    """Merge startup defaults with per-request overrides, then project to
    the keys that drive converter construction. The result is what
    everything downstream uses for cache key + build."""
    merged = dict(_load_startup_settings())
    if settings:
        merged.update(settings)
    return {k: merged.get(k) for k in _BUILD_AFFECTING_KEYS}


def _cache_key(eff_settings: dict[str, Any]) -> str:
    """Stable cache key. JSON with sorted keys handles dicts/lists; we
    don't worry about float NaN because we only ever store JSON-safe vals
    here."""
    return json.dumps(eff_settings, sort_keys=True, default=str)


def get_converter(settings: dict[str, Any] | None = None):
    """Return a DocumentConverter matching the given per-request settings.

    Builds a converter the first time a particular settings combination
    is seen and caches up to _CONVERTER_CACHE_MAX of them (LRU eviction).
    Without this, per-request UI toggles like do_picture_description=true
    would be silently ignored because docling reads them at construction
    time, not per .convert() call."""
    import collections

    global _converter_cache
    if _converter_cache is None:
        _converter_cache = collections.OrderedDict()

    eff = _effective_settings(settings)
    key = _cache_key(eff)

    with _converter_lock:
        if key in _converter_cache:
            _converter_cache.move_to_end(key)
            return _converter_cache[key]
        # Build outside the lock would be ideal, but the lock protects
        # the cache dict — build cost is dominated by HF download which
        # is already cached on disk after the first call.
        converter = _build_converter(settings)
        _converter_cache[key] = converter
        while len(_converter_cache) > _CONVERTER_CACHE_MAX:
            _converter_cache.popitem(last=False)
        return converter


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


def _crops_from_items(
    pictures,
    tables,
    chart_tables_by_id: dict[int, dict[str, Any]] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[int, str]]:
    """Build crops dict + item_id→region_id map for placeholder rewriting.
    Pictures get their classifier sub-label (BAR_CHART, LOGO, …); tables
    are labeled "table". Region IDs are local to the caller (region_1,
    region_2, …) and ordered pictures-then-tables.

    `chart_tables_by_id` is `id(picture) → {gfm, title, raw}` from the
    DePlot pre-pass. When present, the matching crop entry gets a
    `chart_table` field that `_rewrite_picture_tags` will emit beneath
    the image tag in the markdown."""
    chart_tables_by_id = chart_tables_by_id or {}
    crops: dict[str, dict[str, Any]] = {}
    item_to_region: dict[int, str] = {}
    idx = 1
    for picture in pictures:
        rid = f"region_{idx}"
        item_to_region[id(picture)] = rid
        img_attr = getattr(picture, "image", None)
        pil = getattr(img_attr, "pil_image", None) if img_attr is not None else None
        if pil is None:
            continue
        entry: dict[str, Any] = {
            "png": _pil_to_png_bytes(pil),
            "label": _picture_sub_label(picture),
        }
        chart = chart_tables_by_id.get(id(picture))
        if chart and chart.get("gfm"):
            entry["chart_table"] = chart["gfm"]
            if chart.get("title"):
                entry["chart_title"] = chart["title"]
        crops[rid] = entry
        idx += 1
    for table in tables:
        rid = f"region_{idx}"
        item_to_region[id(table)] = rid
        img_attr = getattr(table, "image", None)
        pil = getattr(img_attr, "pil_image", None) if img_attr is not None else None
        if pil is None:
            continue
        crops[rid] = {
            "png": _pil_to_png_bytes(pil),
            "label": "table",
        }
        idx += 1
    return crops, item_to_region


_GRADE_TO_STR = {
    "excellent": "excellent", "good": "good", "fair": "fair", "poor": "poor",
}


def _confidence_to_dict(scores) -> dict[str, Any] | None:
    """Convert a docling PageConfidenceScores / ConfidenceReport into a
    plain JSON-safe dict. Returns None when scores is missing."""
    if scores is None:
        return None
    import math

    def _num(x):
        if x is None:
            return None
        try:
            f = float(x)
            return None if math.isnan(f) else round(f, 4)
        except (TypeError, ValueError):
            return None

    def _grade(g):
        if g is None:
            return None
        s = str(g).split(".")[-1].lower()
        return _GRADE_TO_STR.get(s, s)

    return {
        "parse_score": _num(getattr(scores, "parse_score", None)),
        "layout_score": _num(getattr(scores, "layout_score", None)),
        "table_score": _num(getattr(scores, "table_score", None)),
        "ocr_score": _num(getattr(scores, "ocr_score", None)),
        "mean_score": _num(getattr(scores, "mean_score", None)),
        "low_score": _num(getattr(scores, "low_score", None)),
        "mean_grade": _grade(getattr(scores, "mean_grade", None)),
        "low_grade": _grade(getattr(scores, "low_grade", None)),
    }


def _doc_to_page_result(
    doc,
    page_no: int | None = None,
    page_confidence=None,
    chart_tables_by_id: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Render a single page (or the whole doc if page_no is None) into our
    standard {markdown, crops, width, height, confidence} shape. Both
    pictures and (when generate_table_images is on) tables become crops."""
    # Pictures + tables filtered to this page via prov[0].page_no.
    def _on_page(items):
        if page_no is None:
            return list(items)
        return [
            i for i in items
            if getattr(i, "prov", None) and getattr(i.prov[0], "page_no", None) == page_no
        ]

    pics_on_page = _on_page(getattr(doc, "pictures", []) or [])
    tables_on_page = _on_page(getattr(doc, "tables", []) or [])

    crops, item_to_region = _crops_from_items(
        pics_on_page, tables_on_page, chart_tables_by_id=chart_tables_by_id,
    )

    try:
        markdown = doc.export_to_markdown(page_no=page_no) if page_no is not None \
            else doc.export_to_markdown()
    except Exception:
        markdown = ""

    markdown = _rewrite_picture_tags(markdown, doc, item_to_region, crops)

    # Page dimensions if available.
    w = h = 0
    pages = getattr(doc, "pages", None)
    if pages and page_no in pages:
        size = getattr(pages[page_no], "size", None)
        if size is not None:
            w = int(getattr(size, "width", 0) or 0)
            h = int(getattr(size, "height", 0) or 0)

    return {
        "markdown": markdown,
        "crops": crops,
        "width": w,
        "height": h,
        "confidence": _confidence_to_dict(page_confidence),
    }


def _maybe_extract_chart_tables(doc, settings: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Run DePlot on every picture whose top classification label is a
    chart kind. Returns id(picture) → {gfm, title, raw}. Empty dict when
    do_chart_to_table is False or no chart pictures present.

    Lazy import: keeps `app.chart_extract` (and the Pix2Struct dep) out
    of the API process and unloaded until first opt-in."""
    if not settings.get("do_chart_to_table"):
        return {}
    pictures = getattr(doc, "pictures", []) or []
    if not pictures:
        return {}

    chart_pics: dict[int, Image.Image] = {}
    rid_for_id: dict[int, str] = {}  # for logging only
    for idx, pic in enumerate(pictures, 1):
        label = _picture_sub_label(pic)
        # Local import to avoid circular reference + keep module list slim.
        from . import chart_extract
        if label not in chart_extract.CHART_LABELS:
            continue
        img_attr = getattr(pic, "image", None)
        pil = getattr(img_attr, "pil_image", None) if img_attr is not None else None
        if pil is None:
            continue
        chart_pics[id(pic)] = pil
        rid_for_id[id(pic)] = f"picture_{idx}"

    if not chart_pics:
        return {}

    # Map id-based keys → labels (chart_extract uses strings as keys).
    string_keys = {str(k): img for k, img in chart_pics.items()}
    from . import chart_extract
    parsed = chart_extract.extract_chart_tables(string_keys)
    return {int(sk): parsed[sk] for sk in parsed if sk in parsed}


def extract(image_bytes: bytes, settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """Single-image (or single-page rasterized) extraction.

    Returns {markdown, crops, width, height, confidence}. `confidence` is
    a JSON-safe dict of docling's quality scores (parse/layout/table/ocr
    plus mean/low + grade), or None if scores were unavailable."""
    settings = settings or {}
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size

    converter = get_converter(settings)

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

    chart_tables = _maybe_extract_chart_tables(doc, settings)
    # Single image = one logical page; docling's ConfidenceReport carries
    # overall scores. Use those directly.
    out = _doc_to_page_result(
        doc, page_no=None,
        page_confidence=getattr(result, "confidence", None),
        chart_tables_by_id=chart_tables,
    )
    # For single images we trust the input PIL dimensions over docling's
    # internal page size (which can be in points, not pixels).
    out["width"] = w
    out["height"] = h
    return out


def extract_pdf(pdf_bytes: bytes, settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """Multi-page PDF extraction. Hands the PDF directly to docling so the
    text layer (editable PDFs) is used without per-page OCR.

    Returns {pages: list[per-page result], confidence: overall dict}.
    Each page has the same shape as extract(): {markdown, crops, width,
    height, confidence}. Caller (jobs.run_ocr_pdf) is responsible for
    prefixing region_ids per page.
    """
    settings = settings or {}
    converter = get_converter(settings)

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

    overall = getattr(result, "confidence", None)
    per_page_scores = getattr(overall, "pages", {}) if overall is not None else {}
    chart_tables = _maybe_extract_chart_tables(doc, settings)

    pages = [
        _doc_to_page_result(
            doc, page_no=p,
            page_confidence=per_page_scores.get(p),
            chart_tables_by_id=chart_tables,
        )
        for p in range(1, n_pages + 1)
    ]
    return {"pages": pages, "confidence": _confidence_to_dict(overall)}


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
        tag = f'<image label="{label}">[{rid}]</image>'
        # If DePlot produced a table for this chart, append it as GFM
        # directly under the tag — downstream gets the structured data
        # adjacent to the image reference.
        chart = crops[rid].get("chart_table")
        if chart:
            title = crops[rid].get("chart_title")
            header = f"**Chart data — {title}**\n\n" if title else "**Chart data**\n\n"
            tag = f"{tag}\n\n{header}{chart}"
        return tag

    out = placeholder_re.sub(_sub, markdown).strip()
    return _strip_label_captions(out, crops)


def _strip_label_captions(markdown: str, crops: dict[str, dict[str, Any]]) -> str:
    """Remove docling's standalone-caption paragraphs that duplicate the
    label already encoded in the `<image label="...">` tag.

    Docling's `export_to_markdown` emits the picture classifier label as
    a separate paragraph right after the picture placeholder (e.g.
    "Bar chart", "Logo", "Table"). We've already replaced the placeholder
    with our `<image label="bar_chart">…</image>` form, so that trailing
    caption is pure visual noise. Strip it.

    Algorithm: walk paragraphs. After seeing an `<image>` tag (and through
    any subsequent Chart-data block we ourselves inserted), the *next*
    paragraph whose text equals one of the known captions gets dropped.
    """
    if not crops:
        return markdown

    # Docling builds the caption as label.replace("_"," ").capitalize().
    # We mirror that, with a few common variants for safety.
    captions: set[str] = set()
    for crop in crops.values():
        label = crop.get("label") or ""
        if not label:
            continue
        canon = label.replace("_", " ")
        captions.add(canon.capitalize())
        captions.add(canon.title())

    if not captions:
        return markdown

    paras = markdown.split("\n\n")
    out: list[str] = []
    after_image = False
    for p in paras:
        stripped = p.strip()
        if after_image and stripped in captions:
            after_image = False
            continue
        out.append(p)
        if "<image label=" in stripped:
            after_image = True
        elif after_image and (stripped.startswith("**Chart data") or stripped.startswith("|")):
            # Stay inside the "skip-caption" window across the DePlot block.
            pass
        else:
            after_image = False
    return "\n\n".join(out)
