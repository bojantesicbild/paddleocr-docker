"""FastAPI service: images/PDFs → HillMetrics-compatible markdown."""
from __future__ import annotations

import base64
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from . import markdown_format, ocr_service, pdf_split
from .schemas import ImageCrop, OCRMetadata, OCRResponse

DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parent.parent / "ocr_settings.json"


def _load_default_settings() -> dict[str, Any]:
    try:
        return json.loads(DEFAULT_SETTINGS_PATH.read_text())
    except FileNotFoundError:
        return {}


_DEFAULT_SETTINGS = _load_default_settings()
_READY = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _READY
    # Warm the PP-StructureV3 singleton so first request doesn't pay the model load cost.
    ocr_service.get_pipeline()
    _READY = True
    yield


app = FastAPI(title="PaddleOCR → HillMetrics Markdown", version="0.1.0", lifespan=lifespan)


def _merge_settings(override: str | None) -> dict[str, Any]:
    merged = dict(_DEFAULT_SETTINGS)
    if override:
        try:
            merged.update(json.loads(override))
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"settings must be valid JSON: {e}")
    return merged


def _extract_page(
    image_bytes: bytes,
    settings: dict[str, Any],
    page_number: int,
    region_prefix: str = "",
) -> tuple[str, list[ImageCrop]]:
    result = ocr_service.extract(image_bytes, settings)
    if region_prefix:
        for block in result["blocks"]:
            if block.get("region_id"):
                block["region_id"] = region_prefix + block["region_id"]
        result["crops"] = {region_prefix + k: v for k, v in result["crops"].items()}
    markdown = markdown_format.render_page(result["blocks"], page_number=page_number)
    images: list[ImageCrop] = []
    block_by_region = {b.get("region_id"): b for b in result["blocks"] if b.get("region_id")}
    for region_id, png_bytes in result["crops"].items():
        block = block_by_region.get(region_id, {})
        images.append(
            ImageCrop(
                region_id=region_id,
                page=page_number,
                label=block.get("label", "unknown"),
                bbox=block.get("bbox", [0, 0, 0, 0]),
                png_base64=base64.b64encode(png_bytes).decode("ascii"),
            )
        )
    return markdown, images


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ready" if _READY else "loading"})


@app.post("/ocr/image", response_model=OCRResponse)
async def ocr_image(
    file: UploadFile = File(...),
    page_number: int = Form(1),
    settings: str | None = Form(None),
) -> OCRResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="file must be an image (image/*)")
    merged_settings = _merge_settings(settings)
    image_bytes = await file.read()

    t0 = time.monotonic()
    markdown, images = _extract_page(image_bytes, merged_settings, page_number)
    duration_ms = int((time.monotonic() - t0) * 1000)

    return OCRResponse(
        markdown=markdown,
        images=images,
        metadata=OCRMetadata(
            language=os.environ.get("OCR_LANGUAGE", "fr"),
            page_count=1,
            duration_ms=duration_ms,
            settings=merged_settings,
        ),
    )


@app.post("/ocr/pdf", response_model=OCRResponse)
async def ocr_pdf(
    file: UploadFile = File(...),
    dpi: int = Form(200),
    settings: str | None = Form(None),
) -> OCRResponse:
    if file.content_type and "pdf" not in file.content_type.lower():
        raise HTTPException(status_code=400, detail="file must be a PDF (application/pdf)")
    merged_settings = _merge_settings(settings)
    pdf_bytes = await file.read()

    try:
        total_pages = pdf_split.page_count(pdf_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"could not open PDF: {e}")

    t0 = time.monotonic()
    all_markdown: list[str] = []
    all_images: list[ImageCrop] = []

    for page_idx, page_png in enumerate(pdf_split.split_pdf(pdf_bytes, dpi=dpi), start=1):
        md, imgs = _extract_page(
            page_png, merged_settings, page_idx, region_prefix=f"p{page_idx}_"
        )
        all_markdown.append(md)
        all_images.extend(imgs)

    duration_ms = int((time.monotonic() - t0) * 1000)

    return OCRResponse(
        markdown="\n\n".join(all_markdown),
        images=all_images,
        metadata=OCRMetadata(
            language=os.environ.get("OCR_LANGUAGE", "fr"),
            page_count=total_pages,
            duration_ms=duration_ms,
            settings=merged_settings,
        ),
    )
