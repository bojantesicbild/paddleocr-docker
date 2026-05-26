from typing import Any

from pydantic import BaseModel, Field


class ImageCrop(BaseModel):
    region_id: str
    page: int
    label: str
    bbox: list[float] = Field(description="[x0, y0, x1, y1] in image pixels")
    png_base64: str = Field(description="PNG bytes, base64-encoded")


class OCRMetadata(BaseModel):
    library: str = "docling"
    model: str = "Docling"
    version: str = "2"
    language: str = Field(description="Configured OCR model language (env OCR_LANGUAGE)")
    detected_language: dict[str, Any] | None = Field(
        default=None,
        description="Post-OCR language detection: {code, confidence} or null",
    )
    page_count: int
    duration_ms: int
    confidence: dict[str, Any] | None = Field(
        default=None,
        description="Docling ConfidenceReport: parse/layout/table/ocr scores + mean/low + grade. null if pipeline didn't populate it.",
    )
    settings: dict[str, Any] = Field(default_factory=dict)


class OCRResponse(BaseModel):
    markdown: str
    images: list[ImageCrop]
    metadata: OCRMetadata
