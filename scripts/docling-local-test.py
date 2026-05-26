#!/usr/bin/env python3
"""Local smoke test for the Docling backend.

Runs `app.ocr_service.extract` on a sample image without involving Docker /
OVH / Redis / FastAPI. Verifies the pipeline produces markdown + labelled
crops before we burn a CI cycle.

Usage:
    .venv-docling/bin/python scripts/docling-local-test.py <path-to-image>
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <path-to-image>", file=sys.stderr)
        return 2

    img_path = Path(sys.argv[1])
    if not img_path.is_file():
        print(f"not found: {img_path}", file=sys.stderr)
        return 2

    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)
    os.environ.setdefault("HOME", "/tmp")
    os.environ.setdefault("HF_HOME", str(repo_root / ".venv-docling" / "hf-cache"))

    print(f"loading app.ocr_service (may download ~600-800 MB on first run)...")
    t0 = time.time()
    from app.ocr_service import extract
    print(f"  imported in {time.time() - t0:.1f}s")

    print(f"reading {img_path} ({img_path.stat().st_size // 1024} KB)")
    file_bytes = img_path.read_bytes()
    is_pdf = img_path.suffix.lower() == ".pdf"

    if is_pdf:
        from app.ocr_service import extract_pdf
        print("first PDF inference (cold) — downloads models + warms up...")
        t0 = time.time()
        page_results = extract_pdf(file_bytes, {})
        cold = time.time() - t0
        print(f"  cold run: {cold:.1f}s over {len(page_results)} page(s) "
              f"= {cold / max(1, len(page_results)):.1f}s/page")

        print("second PDF inference (warm)...")
        t0 = time.time()
        page_results = extract_pdf(file_bytes, {})
        warm = time.time() - t0
        print(f"  warm run: {warm:.1f}s over {len(page_results)} page(s) "
              f"= {warm / max(1, len(page_results)):.1f}s/page")

        md = "\n\n".join(r.get("markdown", "") for r in page_results)
        crops = {}
        for page_idx, r in enumerate(page_results, start=1):
            for rid, c in r.get("crops", {}).items():
                crops[f"p{page_idx}_{rid}"] = c
        page_count = len(page_results)
    else:
        print("first inference (cold) — this downloads models + warms up...")
        t0 = time.time()
        result = extract(file_bytes, {})
        cold = time.time() - t0
        print(f"  cold run: {cold:.1f}s")

        print("second inference (warm)...")
        t0 = time.time()
        result = extract(file_bytes, {})
        warm = time.time() - t0
        print(f"  warm run: {warm:.1f}s")

        md = result.get("markdown", "")
        crops = result.get("crops", {})
        page_count = 1

    print("=" * 60)
    print(f"pages:     {page_count}")
    print(f"markdown:  {len(md)} chars, {md.count(chr(10)) + 1} lines")
    print(f"crops:     {len(crops)} regions")
    for rid, c in crops.items():
        size_kib = len(c.get("png", b"")) // 1024
        print(f"  {rid:18s}  label={c.get('label', '?'):20s}  {size_kib} KB")
    print("=" * 60)
    print("markdown (first 2000 chars):")
    print("-" * 60)
    print(md[:2000])
    if len(md) > 2000:
        print(f"... ({len(md) - 2000} more chars)")

    # Persist markdown for inspection
    out_md = Path("/tmp/docling-local-test.md")
    out_md.write_text(md)
    out_json = Path("/tmp/docling-local-test.json")
    out_json.write_text(json.dumps(
        {
            "cold_seconds": cold,
            "warm_seconds": warm,
            "markdown_chars": len(md),
            "crops": {rid: {"label": c.get("label"), "png_kib": len(c.get("png", b"")) // 1024}
                      for rid, c in crops.items()},
        },
        indent=2,
    ))
    print()
    print(f"full markdown saved → {out_md}")
    print(f"summary JSON saved  → {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
