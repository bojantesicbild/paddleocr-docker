"""FastAPI service: images/PDFs → HillMetrics markdown (PaddleOCR-VL-1.5)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import asyncio

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from rq.exceptions import NoSuchJobError
from rq.job import Job, JobStatus

from . import jobs as ocr_jobs
from .schemas import ImageCrop, OCRMetadata, OCRResponse

DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parent.parent / "ocr_settings.json"
INDEX_HTML_PATH = Path(__file__).resolve().parent / "templates" / "index.html"
API_KEY = os.environ.get("OCR_API_KEY", "").strip()


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    """FastAPI dependency. No-op when OCR_API_KEY is unset; otherwise requires
    the header to match."""
    if not API_KEY:
        return
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


def _load_default_settings() -> dict[str, Any]:
    try:
        return json.loads(DEFAULT_SETTINGS_PATH.read_text())
    except FileNotFoundError:
        return {}


_DEFAULT_SETTINGS = _load_default_settings()

app = FastAPI(title="PaddleOCR-VL-1.5 → HillMetrics Markdown", version="0.3.0")


def _merge_settings(override: str | None) -> dict[str, Any]:
    merged = dict(_DEFAULT_SETTINGS)
    if override:
        try:
            merged.update(json.loads(override))
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"settings must be valid JSON: {e}")
    return merged


def _result_to_response(result: dict[str, Any]) -> OCRResponse:
    return OCRResponse(
        markdown=result["markdown"],
        images=[ImageCrop(**img) for img in result["images"]],
        metadata=OCRMetadata(**result["metadata"]),
    )


async def _wait_for_job(job: Job, timeout_s: int = 1800) -> dict[str, Any]:
    """Poll RQ job status until finished/failed. Keeps inference in the worker
    process so the API never loads the VLM itself."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout_s
    while loop.time() < deadline:
        status = job.get_status(refresh=True)
        if status == JobStatus.FINISHED:
            return job.result
        if status == JobStatus.FAILED:
            tail = (job.exc_info or "").splitlines()[-1] if job.exc_info else "job failed"
            raise HTTPException(status_code=500, detail=tail)
        await asyncio.sleep(1)
    raise HTTPException(status_code=504, detail=f"job {job.id} timed out after {timeout_s}s")


@app.get("/config", include_in_schema=False)
def config() -> JSONResponse:
    """Tells the UI whether it needs to send an X-API-Key header."""
    return JSONResponse({"auth_required": bool(API_KEY)})


@app.get("/health")
def health() -> JSONResponse:
    try:
        ok = ocr_jobs.get_redis().ping()
    except Exception:
        ok = False
    return JSONResponse({"status": "ready" if ok else "degraded"})


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(INDEX_HTML_PATH, media_type="text/html")


@app.post("/ocr/image", dependencies=[Depends(require_api_key)])
async def ocr_image(
    file: UploadFile = File(...),
    page_number: int = Form(1),
    settings: str | None = Form(None),
    async_: bool = Query(False, alias="async"),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="file must be an image (image/*)")
    merged = _merge_settings(settings)
    image_bytes = await file.read()

    job = ocr_jobs.get_queue().enqueue(
        ocr_jobs.run_ocr_image,
        image_bytes,
        merged,
        page_number,
        result_ttl=ocr_jobs.RESULT_TTL,
        failure_ttl=ocr_jobs.RESULT_TTL,
    )
    if async_:
        return JSONResponse({"job_id": job.id, "status": "queued"}, status_code=202)
    return _result_to_response(await _wait_for_job(job))


@app.post("/ocr/pdf", dependencies=[Depends(require_api_key)])
async def ocr_pdf(
    file: UploadFile = File(...),
    dpi: int = Form(200),
    settings: str | None = Form(None),
    async_: bool = Query(False, alias="async"),
):
    if file.content_type and "pdf" not in file.content_type.lower():
        raise HTTPException(status_code=400, detail="file must be a PDF (application/pdf)")
    merged = _merge_settings(settings)
    pdf_bytes = await file.read()

    job = ocr_jobs.get_queue().enqueue(
        ocr_jobs.run_ocr_pdf,
        pdf_bytes,
        merged,
        dpi,
        result_ttl=ocr_jobs.RESULT_TTL,
        failure_ttl=ocr_jobs.RESULT_TTL,
    )
    if async_:
        return JSONResponse({"job_id": job.id, "status": "queued"}, status_code=202)
    return _result_to_response(await _wait_for_job(job))


@app.get("/jobs/{job_id}", dependencies=[Depends(require_api_key)])
def job_status(job_id: str):
    try:
        job = Job.fetch(job_id, connection=ocr_jobs.get_redis())
    except NoSuchJobError:
        raise HTTPException(status_code=404, detail="job not found (expired or unknown id)")

    status = job.get_status(refresh=True)
    payload: dict[str, Any] = {
        "job_id": job.id,
        "status": status,
        "enqueued_at": job.enqueued_at.isoformat() if job.enqueued_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "ended_at": job.ended_at.isoformat() if job.ended_at else None,
    }
    if status == JobStatus.FINISHED:
        payload["result"] = job.result
    elif status == JobStatus.FAILED:
        payload["error"] = (job.exc_info or "").splitlines()[-1] if job.exc_info else "job failed"
    return JSONResponse(payload)
