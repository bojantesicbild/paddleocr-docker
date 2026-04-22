"""PaddleOCR-VL-1.5 extraction. Returns VL-native markdown + in-memory crops."""
from __future__ import annotations

import io
import json
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


def _patch_paddle_tensor_int():
    """Workaround for paddle-cpu 3.3.0 bug: Tensor.__int__ breaks on shape-[1]
    tensors with "only 0-dimensional arrays can be converted to Python scalars".
    PaddleOCR-VL's VLM preprocessor hits this. .item() handles both 0-d and
    1-element N-d tensors.
    """
    import paddle  # noqa: WPS433 — deliberate late import inside patcher
    import numpy as np

    def _lenient_int(var):
        arr = np.array(var)
        return int(arr.item() if arr.size == 1 else arr)

    paddle.Tensor.__int__ = _lenient_int


_patch_paddle_tensor_int()

from PIL import Image

_pipe = None
_pipe_lock = threading.Lock()

_SETTINGS_PATH = Path(__file__).resolve().parent.parent / "ocr_settings.json"

# Maps `include_<name>: true|false` settings keys to the PP-DocLayoutV3 label
# that the VL pipeline filters via `markdown_ignore_labels`. A false flag means
# "ignore this label" — consistent with the Baidu demo's behavior where the
# defaults filter all of these out.
_BLOCK_FILTER_FLAGS = {
    "include_header": "header",
    "include_header_image": "header_image",
    "include_footer": "footer",
    "include_footer_image": "footer_image",
    "include_page_number": "number",
    "include_footnote": "footnote",
    "include_aside_text": "aside_text",
}


def _load_startup_settings() -> dict[str, Any]:
    try:
        return json.loads(_SETTINGS_PATH.read_text())
    except FileNotFoundError:
        return {}


def get_pipeline():
    global _pipe
    if _pipe is None:
        with _pipe_lock:
            if _pipe is None:
                from paddleocr import PaddleOCRVL

                s = _load_startup_settings()
                _pipe = PaddleOCRVL(
                    pipeline_version="v1.5",
                    use_doc_orientation_classify=s.get("use_doc_orientation_classify", False),
                    use_doc_unwarping=s.get("use_doc_unwarping", False),
                    use_chart_recognition=s.get("chart_recognition", False),
                    use_seal_recognition=s.get("use_seal_recognition", False),
                    # The multi-process VLM worker pool crashes on CPU paddle
                    # ("only 0-dimensional arrays can be converted to Python scalars").
                    # Disable — we're single-request-at-a-time anyway.
                    use_queues=False,
                )
    return _pipe


def _pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    if img.mode != "RGB" and img.mode != "RGBA":
        img = img.convert("RGB")
    img.save(buf, format="PNG")
    return buf.getvalue()


_MD_IMG_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def extract(image_bytes: bytes, settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run PaddleOCR-VL-1.5 on a single page image.

    Returns dict with:
      markdown: final markdown (image refs already rewritten to HillMetrics tags)
      crops: {region_id: png_bytes}
      width, height: page dimensions
    """
    settings = settings or {}
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size

    pipe = get_pipeline()

    predict_kwargs = {
        "use_doc_orientation_classify": settings.get("use_doc_orientation_classify"),
        "use_doc_unwarping": settings.get("use_doc_unwarping"),
        "use_chart_recognition": settings.get("chart_recognition"),
        "use_seal_recognition": settings.get("use_seal_recognition"),
        "use_ocr_for_image_block": settings.get("use_ocr_for_image_block"),
        "layout_threshold": settings.get("layout_threshold"),
        "use_queues": False,
    }
    predict_kwargs = {k: v for k, v in predict_kwargs.items() if v is not None}

    # Build the ignore-label list from the include_* flags. Anything not
    # explicitly included is filtered out of the final markdown.
    ignore_labels = [
        label for key, label in _BLOCK_FILTER_FLAGS.items() if not settings.get(key, False)
    ]
    predict_kwargs["markdown_ignore_labels"] = ignore_labels

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        img_path = tmp / "page.png"
        img.save(img_path, format="PNG")
        results = list(pipe.predict(str(img_path), **predict_kwargs))

    if not results:
        return {"markdown": "", "crops": {}, "width": w, "height": h}

    r = results[0]
    md_info = r._to_markdown(pretty=False)
    raw_markdown = md_info.get("markdown_texts") or ""
    md_images: dict[str, Image.Image] = md_info.get("markdown_images") or {}

    # Build path → label map by walking the VL pipeline's per-block structure.
    # Blocks with images expose block.image["path"], matched against the paths
    # embedded in markdown's ![](path) refs.
    path_to_label: dict[str, str] = {}
    for block in r.get("parsing_res_list", []):
        img = getattr(block, "image", None)
        if img and img.get("path"):
            label = (getattr(block, "label", "") or "image").lower()
            path_to_label[img["path"]] = label

    # Assign stable region IDs in path-order as they appear in markdown
    path_to_region: dict[str, str] = {}
    for idx, path in enumerate(_MD_IMG_RE.findall(raw_markdown), start=1):
        path_to_region.setdefault(path, f"region_{idx}")

    crops: dict[str, dict[str, Any]] = {}
    for path, region_id in path_to_region.items():
        pil = md_images.get(path)
        if pil is None:
            continue
        crops[region_id] = {
            "png": _pil_to_png_bytes(pil),
            "label": path_to_label.get(path, "image"),
        }

    # Rewrite ![…](path) → <image label="…">[region_id]</image>
    def _replace(m: re.Match) -> str:
        path = m.group(1)
        rid = path_to_region.get(path)
        if not rid:
            return m.group(0)
        label = path_to_label.get(path, "image")
        return f'<image label="{label}">[{rid}]</image>'

    markdown = _MD_IMG_RE.sub(_replace, raw_markdown).strip()

    return {
        "markdown": markdown,
        "crops": crops,
        "width": w,
        "height": h,
    }
