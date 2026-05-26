"""Chart-to-table extraction using google/deplot (Pix2Struct-1.3B).

Docling's built-in `do_chart_extraction` uses Granite-Vision-V4 which (a) hits
a transformers ↔ remote-code signature skew on the cached revision and (b)
forces flash_attention_2 — unsupported on V100 (CC 7.0). DePlot is a
purpose-built plot-to-table model that runs in plain transformers and
doesn't need FA2.

Wire:
  - lazy singleton (model + processor)
  - input: PIL chart crops + their region_ids
  - output: dict region_id → GFM markdown table string
  - inserted into the markdown right after the `<image label="bar_chart">…`
    tag so downstream consumers see numerical data alongside the picture.

Trade-off: ~1.3B params (~5 GB FP16 on CUDA), ~3-5 s/chart on V100. Off by
default — opt in via settings `do_chart_to_table=True`.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any

from PIL import Image

log = logging.getLogger(__name__)

_MODEL_ID = "google/deplot"
_PROMPT = "Generate underlying data table of the figure below:"

_model = None
_processor = None
_device = None
_lock = threading.Lock()

# Picture-classifier labels we treat as charts. DocumentFigureClassifier-v2.5
# uses these exact strings; we lowercase the comparison just in case.
CHART_LABELS = frozenset({
    "bar_chart",
    "line_chart",
    "pie_chart",
    "scatter_plot",
    "histogram",
    "area_chart",
    "stacked_bar_chart",
})


def _select_device() -> str:
    """CUDA when available, MPS skipped (DePlot's Pix2Struct trips on
    float64 ops MPS can't run). CPU fallback for local dev."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _load() -> tuple[Any, Any, str]:
    """Idempotent model load. Returns (model, processor, device)."""
    global _model, _processor, _device
    if _model is not None and _processor is not None:
        return _model, _processor, _device  # type: ignore[return-value]
    with _lock:
        if _model is None or _processor is None:
            from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
            import torch

            _device = _select_device()
            dtype = torch.float16 if _device == "cuda" else torch.float32
            log.info("loading DePlot (%s, dtype=%s) — ~5 GB on CUDA", _MODEL_ID, dtype)
            m = Pix2StructForConditionalGeneration.from_pretrained(_MODEL_ID, torch_dtype=dtype)
            if _device == "cuda":
                m = m.to("cuda")
            m.eval()
            _model = m
            _processor = Pix2StructProcessor.from_pretrained(_MODEL_ID)
    return _model, _processor, _device  # type: ignore[return-value]


def _parse_deplot_to_gfm(raw: str) -> tuple[str | None, str]:
    """DePlot emits one logical row per `<0x0A>` marker. Layout:

        TITLE | <chart title> <0x0A> | h1 | h2 <0x0A> r1c1 | r1c2 | r1c3 …

    Title chunk is optional. First non-title chunk is the header; rest are
    data rows. Cells are pipe-delimited; some rows have an empty leading
    cell (artifact of how DePlot tokenizes the table-start)."""
    parts = [p.strip() for p in raw.split("<0x0A>") if p.strip()]
    title: str | None = None
    if parts and parts[0].upper().startswith("TITLE"):
        # "TITLE | <text>"
        head = parts.pop(0)
        if "|" in head:
            title = head.split("|", 1)[1].strip() or None
    if not parts:
        return title, ""

    rows: list[list[str]] = []
    for p in parts:
        cells = [c.strip() for c in p.split("|")]
        # Drop leading empties (DePlot's "| h1 | h2" pattern produces one).
        while cells and cells[0] == "":
            cells.pop(0)
        if cells:
            rows.append(cells)
    if not rows:
        return title, ""

    ncols = max(len(r) for r in rows)
    header = rows[0] + [""] * (ncols - len(rows[0]))
    out: list[str] = ["| " + " | ".join(header) + " |", "|" + "---|" * ncols]
    for r in rows[1:]:
        cells = (r + [""] * ncols)[:ncols]
        out.append("| " + " | ".join(cells) + " |")
    return title, "\n".join(out)


def extract_chart_tables(
    chart_images: dict[str, Image.Image],
    max_new_tokens: int = 512,
) -> dict[str, dict[str, Any]]:
    """Run DePlot over a dict of region_id → PIL.Image (chart crops).

    Returns region_id → {"gfm": str (markdown table or empty), "title":
    str|None, "raw": str}. Empty crops or per-chart failures are caught
    and skipped — one bad chart doesn't tank the pipeline."""
    if not chart_images:
        return {}

    model, processor, device = _load()
    import torch

    results: dict[str, dict[str, Any]] = {}
    # Match the model's dtype on float tensors only. Pix2StructProcessor
    # returns flattened_patches as float32 by default — feeding that into
    # an fp16 model crashes with "mat1 and mat2 dtype mismatch" inside the
    # patch projection.
    model_dtype = next(model.parameters()).dtype
    # One chart at a time — DePlot's processor handles a list but the
    # per-image input shapes differ, batching gives no win without
    # padding/cropping hacks. ~3-5s/chart on V100 is fine.
    for rid, img in chart_images.items():
        try:
            if img.mode != "RGB":
                img = img.convert("RGB")
            inputs = processor(images=img, text=_PROMPT, return_tensors="pt")
            cast_inputs: dict[str, Any] = {}
            for k, v in inputs.items():
                if device == "cuda":
                    v = v.to("cuda")
                # Only cast floating-point tensors; integer ids/masks stay
                # as int64.
                if v.is_floating_point():
                    v = v.to(model_dtype)
                cast_inputs[k] = v
            with torch.inference_mode():
                out_ids = model.generate(**cast_inputs, max_new_tokens=max_new_tokens)
            raw = processor.decode(out_ids[0], skip_special_tokens=True)
            title, gfm = _parse_deplot_to_gfm(raw)
            results[rid] = {"gfm": gfm, "title": title, "raw": raw}
        except Exception as e:  # noqa: BLE001
            # First-chart-only: re-raise so the worker traceback surfaces
            # via /jobs/{id}. Subsequent chart failures get logged + null'd
            # so one bad chart doesn't kill the rest. Switch the bare
            # `raise` to the swallow path once DePlot is known-working on
            # the target GPU.
            log.warning("DePlot failed on %s: %s", rid, e)
            if not results:
                raise
            results[rid] = {"gfm": "", "title": None, "raw": "", "error": str(e)}
    return results
