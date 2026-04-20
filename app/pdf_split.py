from collections.abc import Iterator

import fitz


def split_pdf(pdf_bytes: bytes, dpi: int = 200) -> Iterator[bytes]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    scale = dpi / 72.0
    mat = fitz.Matrix(scale, scale)
    try:
        for page in doc:
            pix = page.get_pixmap(matrix=mat)
            yield pix.tobytes("png")
    finally:
        doc.close()


def page_count(pdf_bytes: bytes) -> int:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return doc.page_count
    finally:
        doc.close()
