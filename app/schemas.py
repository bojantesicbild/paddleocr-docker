from typing import Any

from pydantic import BaseModel, Field


class ImageCrop(BaseModel):
    region_id: str
    page: int
    label: str
    bbox: list[float] = Field(description="[x0, y0, x1, y1] in image pixels")
    png_base64: str = Field(description="PNG bytes, base64-encoded")


class OCRMetadata(BaseModel):
    library: str = "paddleocr"
    model: str = "PaddleOCR-VL-1.5"
    version: str = "3.4.0"
    language: str
    page_count: int
    duration_ms: int
    settings: dict[str, Any] = Field(default_factory=dict)


class OCRResponse(BaseModel):
    markdown: str
    images: list[ImageCrop]
    metadata: OCRMetadata
