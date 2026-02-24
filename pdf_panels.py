#!/usr/bin/env python3
"""
Extract storyboard panel images directly from PDF structure.

Reads image placement coordinates from the PDF (as written by InDesign,
PowerPoint, etc.) and returns cropped images in reading order.

This is a DUMB image extractor — it does NOT attempt triptych detection,
caption parsing, or frame number extraction. Claude Vision handles all
structural decisions (which images form pan/tilt/track sequences, etc.).

Usage:
  python3 pdf_panels.py <pdf_path> <page_number> [min_width] [min_height]

Returns JSON:
  {
    "count": 8,
    "mode": "pdf_structure",
    "images": [
      { "x": 15.4, "y": 153.7, "width": 249.6, "height": 107.4, "image": "<base64>" },
      ...
    ],
    "pageWidth": 612.0,
    "pageHeight": 792.0
  }
"""
import fitz
import json
import sys
import base64


def render_panel(page, bbox, mat):
    """Render a single panel region to base64 JPEG."""
    rect = fitz.Rect(bbox)
    pix = page.get_pixmap(matrix=mat, clip=rect)
    img_bytes = pix.tobytes("jpeg")
    return base64.b64encode(img_bytes).decode("ascii")


def images_overlap(a, b, threshold=0.3):
    """Check if two bboxes overlap by more than threshold of the smaller area."""
    x_overlap = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    y_overlap = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = x_overlap * y_overlap
    if intersection == 0:
        return False
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    smaller_area = min(area_a, area_b)
    return (intersection / smaller_area) > threshold if smaller_area > 0 else False


def extract_panels(pdf_path, page_num, min_w=100, min_h=50, zoom=3):
    """
    Extract image bboxes from a PDF page and return cropped images in reading order.
    Does NOT attempt triptych detection or caption parsing.
    """
    doc = fitz.open(pdf_path)

    if page_num < 1 or page_num > len(doc):
        return {"count": 0, "mode": "pdf_structure", "images": [],
                "error": f"Page {page_num} out of range"}

    page = doc[page_num - 1]
    page_width = page.rect.width
    page_height = page.rect.height

    blocks = page.get_text("dict")["blocks"]
    img_blocks = [b for b in blocks if b["type"] == 1]

    # Filter: real panels vs small overlays/icons
    raw_panels = []
    skipped = []
    for b in img_blocks:
        bbox = b["bbox"]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w >= min_w and h >= min_h:
            raw_panels.append(bbox)
        else:
            skipped.append({"w": round(w, 1), "h": round(h, 1),
                            "x": round(bbox[0], 1), "y": round(bbox[1], 1)})

    if skipped:
        print(f"[PDF] Page {page_num}: skipped {len(skipped)} small images: {skipped}",
              file=sys.stderr, flush=True)

    if not raw_panels:
        doc.close()
        return {"count": 0, "mode": "pdf_structure", "images": [],
                "pageWidth": round(page_width, 1), "pageHeight": round(page_height, 1)}

    # Merge overlapping images — some visual frames are composed of multiple
    # PDF image objects (base photo + text overlay + product shot composited).
    # Keep only the larger one when overlap > threshold.
    merged_panels = []
    used = set()
    raw_panels.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    for i, panel_a in enumerate(raw_panels):
        if i in used:
            continue
        for j, panel_b in enumerate(raw_panels):
            if j <= i or j in used:
                continue
            if images_overlap(panel_a, panel_b):
                used.add(j)
        merged_panels.append(panel_a)

    if len(merged_panels) < len(raw_panels):
        print(f"[PDF] Page {page_num}: merged {len(raw_panels)} overlapping images -> {len(merged_panels)} panels",
              file=sys.stderr, flush=True)
    raw_panels = merged_panels

    # Sort into reading order: group by rows, then left-to-right within each row
    raw_panels.sort(key=lambda b: b[1])
    row_threshold = 30
    rows = []
    current_row = [raw_panels[0]]
    for p in raw_panels[1:]:
        if abs(p[1] - current_row[0][1]) < row_threshold:
            current_row.append(p)
        else:
            rows.append(sorted(current_row, key=lambda b: b[0]))
            current_row = [p]
    rows.append(sorted(current_row, key=lambda b: b[0]))

    # Render all images in reading order
    mat = fitz.Matrix(zoom, zoom)
    result_images = []

    for row in rows:
        for bbox in row:
            b64 = render_panel(page, bbox, mat)
            result_images.append({
                "x": round(bbox[0], 1),
                "y": round(bbox[1], 1),
                "width": round(bbox[2] - bbox[0], 1),
                "height": round(bbox[3] - bbox[1], 1),
                "image": b64
            })

    doc.close()

    print(f"[PDF] Page {page_num}: {len(result_images)} images extracted from PDF structure",
          file=sys.stderr, flush=True)

    return {
        "count": len(result_images),
        "mode": "pdf_structure",
        "images": result_images,
        "pageWidth": round(page_width, 1),
        "pageHeight": round(page_height, 1)
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: pdf_panels.py <pdf_path> <page_number> [min_width] [min_height]"}))
        sys.exit(1)

    pdf_path = sys.argv[1]
    page_num = int(sys.argv[2])
    min_w = float(sys.argv[3]) if len(sys.argv) > 3 else 100
    min_h = float(sys.argv[4]) if len(sys.argv) > 4 else 50

    try:
        result = extract_panels(pdf_path, page_num, min_w, min_h)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
