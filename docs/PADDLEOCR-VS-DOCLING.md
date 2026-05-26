# PaddleOCR vs Docling

Short decision-support note for picking a document-parsing backend.
Captures what we learned running both on an OVH V100S in this repo.

## TL;DR

| Question | Answer |
|---|---|
| Best output quality on dense / messy pages | **PaddleOCR-VL-1.5** (vision-language model) |
| Best speed on V100-class GPUs (CC 7.0) | **Docling** by 10–20× |
| Best speed on Ampere or newer (CC 8.0+) with vLLM | PaddleOCR-VL-1.5 |
| Richer per-region labels for routing (chart vs logo vs signature) | **Docling** (26 sub-types) |
| Reads text *inside* charts/diagrams natively | **PaddleOCR-VL-1.5** |
| Pure-CPU usable (no GPU) | **Docling** (smaller models) |
| Maturity / single-vendor backing | PaddleOCR: Baidu · Docling: IBM Research |

For this repo we ended up on Docling because the V100S we deploy on
is one GPU generation too old for the modern fast-inference stack
(vLLM / FlashAttention 2 require CC ≥ 8.0).

## Architecture

### PaddleOCR-VL-1.5
A single ~0.9 B parameter vision-language model. Layout, OCR, table
structure, formula reading, and chart-text reading are all done by
one autoregressive decoder. The model literally "reads" the page like
a small Qwen-VL.

End-to-end → high quality, especially on dense and unusual layouts.
Trade-off: every region's content is generated token by token, which
is slow without a modern inference engine (vLLM, FlashAttention).

### Docling
A pipeline of small specialised models orchestrated by a Python SDK:

| Stage | Model |
|---|---|
| Page layout | DocLing layout model (~100 MB) |
| Table structure | TableFormer (~200 MB) |
| OCR (text inside regions) | EasyOCR / RapidOCR / Tesseract (~100–200 MB) |
| Picture sub-classification | DocumentFigureClassifier v2.5 (~4 M params) |
| Optional caption | a VLM of your choice |

Each model is non-autoregressive and runs in parallel where possible.
No generation step in the hot path → much less GPU dependency.

## Output labels

### PaddleOCR-VL — flat label set
`paragraph_title`, `doc_title`, `text`, `table`, `chart`, `image`,
`seal`, `formula`, `header`, `header_image`, `footer`,
`footer_image`, `footnote`, `number` (page number), `aside_text`, …

One level. `chart` exists separately from `image`, but you can't
tell what *kind* of chart without doing extra work yourself.

### Docling — two layers

**Layer 1 (`DocItemLabel`)** — coarse type, on every region:
```
CAPTION · CHART · FOOTNOTE · FORMULA · LIST_ITEM · PAGE_FOOTER
PAGE_HEADER · PICTURE · SECTION_HEADER · TABLE · TEXT · TITLE
DOCUMENT_INDEX · CODE · CHECKBOX_SELECTED · CHECKBOX_UNSELECTED
FORM · KEY_VALUE_REGION · GRADING_SCALE · HANDWRITTEN_TEXT
PARAGRAPH · REFERENCE · FIELD_* · MARKER
```

**Layer 2 (`PictureClassificationLabel`)** — sub-type for items
labelled PICTURE, via the DocumentFigureClassifier-v2.5 model:
```
BAR_CHART · LINE_CHART · PIE_CHART · SCATTER_CHART · SCATTER_PLOT
STACKED_BAR_CHART · BOX_PLOT · FLOW_CHART · HEATMAP · STRATIGRAPHIC_CHART
TABLE (image-of-table) · FULL_PAGE_IMAGE · PAGE_THUMBNAIL
PHOTOGRAPH · NATURAL_IMAGE · SCREENSHOT · SCREENSHOT_FROM_COMPUTER
SCREENSHOT_FROM_MANUAL · LOGO · SIGNATURE · STAMP · ICON
BAR_CODE · QR_CODE · GEOGRAPHIC_MAP · TOPOGRAPHICAL_MAP · REMOTE_SENSING
CHEMISTRY_STRUCTURE · MOLECULAR_STRUCTURE · MARKUSH_STRUCTURE
ENGINEERING_DRAWING · CAD_DRAWING · ELECTRICAL_DIAGRAM
CALENDAR · CROSSWORD_PUZZLE · MUSIC · PICTURE_GROUP · OTHER
```

This is the killer feature for downstream routing — you can send a
`BAR_CHART` to a chart-aware reader, `LOGO` to a logo classifier,
`SIGNATURE` to a separate flow, all from the layout step.

## Speed on V100S (CC 7.0)

Measured on the same dense financial-report page (~A4, ~30 blocks):

| Backend | First request | Steady state |
|---|---|---|
| PaddleOCR-VL-1.5 (native paddle) | ~600 s (cold) | ~114 s/page |
| PaddleOCR-VL-1.5 (vLLM) | — | not supported on CC 7.0 |
| Docling | ~30–60 s (model download + load) | ~5–15 s/page (target) |
| Baidu's hosted demo (A100 + vLLM) | — | ~4–5 s/page |

PaddleOCR-VL on V100 is **slow because the VLM has to generate every
region token by token**, and V100 has no FlashAttention-2 support
(SM70 < SM80 required).

## Speed on A100 / H100 / L4 / A10 (CC 8.0+)

PaddleOCR-VL closes the gap because vLLM and FlashAttention 2 are
finally available. Expect **5–15 s/page** with the official
`paddleocr genai_server --backend vllm` setup.

## Memory footprint

| Backend | Image size | Model weights | VRAM at idle |
|---|---|---|---|
| PaddleOCR-VL-1.5 | ~14 GB | ~5 GB | ~3.6 GB allocated, ~9 GB reserved |
| Docling | ~6 GB | ~0.6–0.8 GB | depends on enrichments |

## When to pick which

**Pick PaddleOCR-VL-1.5 when:**
- You have a modern GPU (CC ≥ 8.0)
- Documents have weird layouts the layout-detector might mis-parse
- You need to read text *inside* charts/diagrams (axis labels, legend)
- Quality is more important than speed

**Pick Docling when:**
- You're on older GPUs (V100, T4) or CPU
- You need fine-grained picture labels for downstream routing
- You want a smaller image and faster cold starts
- 1–2 s of generation latency is unacceptable

## Costs

Both are MIT/Apache licensed, self-hostable, no per-call fees.
PaddleOCR's models are on Hugging Face under `PaddlePaddle/`; Docling's
under `ds4sd/` and `docling-project/`.

## In this repo

`main` runs Docling on the OVH V100S. To switch back to PaddleOCR-VL
we'd revert the merge commit that swapped backends (the codebase
keeps the same external API contract — same `<image label="…">[region_N]</image>`
tags, same JSON shape, same queue/UI/auth/deploy plumbing). Both
backends populate `metadata.detected_language` via `langdetect`, both
return base64-encoded crops with `region_id` + `label`.

## References

- PaddleOCR-VL-1.5 paper: <https://arxiv.org/abs/2510.14528>
- Docling paper: <https://arxiv.org/abs/2408.09869>
- Docling enrichments docs: <https://docling-project.github.io/docling/usage/enrichments/>
- DocumentFigureClassifier-v2.5: <https://huggingface.co/docling-project/DocumentFigureClassifier-v2.5>
- PaddleOCR-VL backend / hardware matrix: <https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html>
