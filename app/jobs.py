"""Queue wiring + job functions.

Redis is a local supervisord-managed process on 127.0.0.1:6379 inside the
container. One queue, one worker — inference is serialized by design (the
VLM is GPU-bound and OOM-prone when overlapped).
"""
from __future__ import annotations

import base64
import os
from typing import Any

from redis import Redis
from rq import Queue

from . import markdown_format  # cheap; no heavy backend imports

# Seed langdetect's internal RNG for reproducible results across calls.
from langdetect import DetectorFactory, LangDetectException, detect_langs

DetectorFactory.seed = 0

# ocr_service and pdf_split are imported lazily inside the job functions so
# the API process never loads torch / docling (only the worker does).

REDIS_URL = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")
QUEUE_NAME = "ocr"
JOB_TIMEOUT = int(os.environ.get("OCR_JOB_TIMEOUT", "1800"))  # 30 min
RESULT_TTL = int(os.environ.get("OCR_RESULT_TTL", "86400"))   # 24 h

_redis: Redis | None = None
_queue: Queue | None = None


def get_redis() -> Redis:
    global _redis
    if _redis is None:
        _redis = Redis.from_url(REDIS_URL)
    return _redis


def get_queue() -> Queue:
    global _queue
    if _queue is None:
        _queue = Queue(
            QUEUE_NAME,
            connection=get_redis(),
            default_timeout=JOB_TIMEOUT,
        )
    return _queue


def _detect_language(markdown: str) -> dict[str, Any] | None:
    """Run langdetect on the final markdown. Returns {'code': 'fr',
    'confidence': 0.97} or None if detection fails (too little text, all
    symbols, etc.)."""
    if len(markdown.strip()) < 20:
        return None
    try:
        candidates = detect_langs(markdown)
    except LangDetectException:
        return None
    if not candidates:
        return None
    top = candidates[0]
    return {"code": top.lang, "confidence": round(top.prob, 4)}


def _page_to_payload(extract_result: dict[str, Any], page_number: int, region_prefix: str = "") -> dict[str, Any]:
    """Shape a single-page extract_result into the API response contract (markdown + images)."""
    markdown = extract_result["markdown"]
    crops: dict[str, dict[str, Any]] = extract_result["crops"]

    markdown = markdown_format.html_tables_to_gfm(markdown)
    markdown = markdown_format.collapse_blank_lines(markdown)

    if region_prefix:
        markdown, mapping = markdown_format.prefix_region_ids(markdown, region_prefix)
        crops = {mapping.get(k, k): v for k, v in crops.items()}

    markdown = markdown_format.render_page(markdown, page_number=page_number)

    images = [
        {
            "region_id": rid,
            "page": page_number,
            "label": crop.get("label", "image"),
            "bbox": [0, 0, 0, 0],
            "png_base64": base64.b64encode(crop["png"]).decode("ascii"),
        }
        for rid, crop in crops.items()
    ]
    return {"markdown": markdown, "images": images}


def run_ocr_image(image_bytes: bytes, settings: dict[str, Any], page_number: int) -> dict[str, Any]:
    import time
    from . import ocr_service
    t0 = time.monotonic()
    res = ocr_service.extract(image_bytes, settings)
    payload = _page_to_payload(res, page_number)
    return {
        "markdown": payload["markdown"],
        "images": payload["images"],
        "metadata": {
            "library": "docling",
            "model": "Docling",
            "version": "2",
            "language": os.environ.get("OCR_LANGUAGE", "ch"),
            "detected_language": _detect_language(payload["markdown"]),
            "page_count": 1,
            "duration_ms": int((time.monotonic() - t0) * 1000),
            "settings": settings,
        },
    }


def run_ocr_pdf(pdf_bytes: bytes, settings: dict[str, Any], dpi: int) -> dict[str, Any]:
    """Hand the whole PDF to docling so editable text layers are used
    without per-page OCR. `dpi` kept in the signature for API stability
    but no longer drives anything (docling rasterizes internally only
    where needed)."""
    import time
    from . import ocr_service
    t0 = time.monotonic()

    page_results = ocr_service.extract_pdf(pdf_bytes, settings)
    total_pages = len(page_results)

    all_markdown: list[str] = []
    all_images: list[dict[str, Any]] = []
    for page_idx, res in enumerate(page_results, start=1):
        payload = _page_to_payload(res, page_idx, region_prefix=f"p{page_idx}_")
        all_markdown.append(payload["markdown"])
        all_images.extend(payload["images"])

    full_markdown = "\n\n".join(all_markdown)
    return {
        "markdown": full_markdown,
        "images": all_images,
        "metadata": {
            "library": "docling",
            "model": "Docling",
            "version": "2",
            "language": os.environ.get("OCR_LANGUAGE", "ch"),
            "detected_language": _detect_language(full_markdown),
            "page_count": total_pages,
            "duration_ms": int((time.monotonic() - t0) * 1000),
            "settings": settings,
        },
    }
