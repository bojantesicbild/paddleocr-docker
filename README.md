# paddleocr-docker

Dockerized PaddleOCR (PP-StructureV3) HTTP service. Accepts images or PDFs and
returns Markdown aligned with the HillMetrics.AI OCR contract
(`> Page N` blockquote page markers, GitHub-flavored tables,
`<image>[region_id]</image>` placeholders, `<logo>…</logo>` for seals), plus
base64-encoded crops for downstream vision-LLM description.

## Stack

- FastAPI + Uvicorn
- PaddleOCR 3.4.0 — PP-StructureV3 pipeline
- PyMuPDF for PDF → page images
- markdownify for HTML table → GFM

## Endpoints

| Method | Path         | Body                                                       | Returns |
|--------|--------------|------------------------------------------------------------|---------|
| GET    | `/health`    | —                                                          | `{"status":"ready"}` once the model is loaded |
| POST   | `/ocr/image` | multipart: `file` (image), `page_number` (int, default 1), `settings` (JSON string, optional) | `OCRResponse` |
| POST   | `/ocr/pdf`   | multipart: `file` (PDF), `dpi` (int, default 200), `settings` (JSON string, optional) | `OCRResponse` |

### `OCRResponse` shape

```json
{
  "markdown": "> Page 1\n\n# Title\n\nBody text…\n\n<image>[region_3]</image>\n\n| a | b |\n| --- | --- |\n| 1 | 2 |",
  "images": [
    {
      "region_id": "region_3",
      "page": 1,
      "label": "chart",
      "bbox": [x0, y0, x1, y1],
      "png_base64": "iVBORw0KGgoAAAA…"
    }
  ],
  "metadata": {
    "library": "paddleocr",
    "model": "PP-StructureV3",
    "version": "3.4.0",
    "language": "ch",
    "page_count": 1,
    "duration_ms": 1234,
    "settings": { "use_table_recognition": true, "layout_threshold": 0.3 }
  }
}
```

For multi-page PDFs the `region_id` values are prefixed with `p{N}_` so they're
unique across pages and stay consistent with the `<image>[…]</image>`
placeholders embedded in the markdown.

## Build & run (Docker)

```bash
docker compose build        # ~10–15 min first time (wheels + model download baked into image)
docker compose up           # listens on http://localhost:8080
curl -F file=@sample.pdf -F dpi=200 http://localhost:8080/ocr/pdf | jq .metadata
```

The image is ~2 GB because the PP-StructureV3 weights are pre-cached during
build so the container has a fast cold start and works offline.

## Configuration

- `OCR_LANGUAGE` (env var) — PaddleOCR `lang` code. Default `ch` (multilingual
  PP-OCRv5, covers CJK + Latin). Override with `fr`, `en`, etc. if you want a
  language-specific model.
- `ocr_settings.json` — runtime flags loaded at startup (table/formula/seal
  toggles, layout threshold). Per-request overrides via the `settings` form field.
- `pipeline_config.yaml` — PP-StructureV3 sub-pipeline configuration (model
  names, thresholds, NMS, merge modes). Used as-is by PaddleOCR.

## Local development (without Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OCR_LANGUAGE=ch
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

First request takes ~1–2 min on CPU while PP-StructureV3 warms up.

## Layout

```
app/
├── main.py             # FastAPI app + endpoints
├── ocr_service.py      # PP-StructureV3 singleton + block/crop extraction
├── markdown_format.py  # PP-StructureV3 labels → HillMetrics markdown
├── pdf_split.py        # PyMuPDF PDF → PNG pages (configurable DPI)
└── schemas.py          # Pydantic models (OCRResponse, ImageCrop, OCRMetadata)
Dockerfile              # multi-stage, CPU, models baked in
docker-compose.yml      # local dev / single-service deploy
pipeline_config.yaml    # PP-StructureV3 configuration
ocr_settings.json       # default OCR flags
requirements.txt
```
