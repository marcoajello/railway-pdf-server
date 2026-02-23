#!/usr/bin/env python3
"""
Extract storyboard panel images directly from PDF structure.

Instead of rendering the page and trying to find panels with computer vision,
this reads the image placement coordinates that the layout tool (InDesign,
PowerPoint, etc.) already wrote into the PDF.

Usage:
  python3 pdf_panels.py <pdf_path> <page_number> [min_width] [min_height]

Returns JSON:
  {
    "count": 6,
    "mode": "pdf_structure",
    "panels": [
      { "x": 15.4, "y": 153.7, "width": 249.6, "height": 107.4, "image": "<base64>", "caption": "..." },
      ...
    ]
  }

Panels are sorted in reading order (top-to-bottom, left-to-right).
Small overlay images (icons, stat badges) are filtered by minimum size.
Triptych/pan sequences (3 images sharing one caption) are merged into single panels.
"""
import fitz
import json
import sys
import base64
import io


def extract_caption(text_block):
    """Extract formatted text from a PDF text block."""
    lines = []
    for line in text_block["lines"]:
        line_text = ""
        for span in line["spans"]:
            text = span["text"]
            if not text.strip():
                line_text += text
                continue
            bold = bool(span["flags"] & 16)
            italic = bool(span["flags"] & 2)
            if bold and italic:
                line_text += f"<b><i>{text}</i></b>"
            elif bold:
                line_text += f"<b>{text}</b>"
            elif italic:
                line_text += f"<i>{text}</i>"
            else:
                line_text += text
        line_text = line_text.strip()
        if line_text:
            lines.append(line_text)
    return "\n".join(lines)


def find_caption_for_region(img_left, img_right, img_bottom, text_blocks, max_y_dist=60):
    """Find the caption text block below an image region."""
    best = None
    best_dist = 999
    for tb in text_blocks:
        tb_bbox = tb["bbox"]
        y_dist = tb_bbox[1] - img_bottom
        h_overlap = min(tb_bbox[2], img_right) - max(tb_bbox[0], img_left)
        if -5 <= y_dist < max_y_dist and h_overlap > 30 and y_dist < best_dist:
            best = tb
            best_dist = y_dist
    return best


def count_captions_for_row(row_panels, row_bottom, text_blocks):
    """
    Count how many separate caption text blocks sit below a row of images.
    Returns list of unique caption blocks found.
    """
    seen = []
    for bbox in row_panels:
        cap = find_caption_for_region(bbox[0], bbox[2], row_bottom, text_blocks)
        if cap is not None and id(cap) not in [id(s) for s in seen]:
            seen.append(cap)
    return seen


def extract_panels(pdf_path, page_num, min_w=100, min_h=50, zoom=3):
    """
    Extract panel images from a PDF page using its structural metadata.
    
    1. Read image bounding boxes from PDF structure (get_text dict)
    2. Filter out small overlays by minimum size
    3. Detect triptych/pan sequences (3 images sharing 1 caption) and merge
    4. Render each panel region at high resolution
    5. Return sorted panels with base64 JPEG data and formatted captions
    """
    doc = fitz.open(pdf_path)
    
    if page_num < 1 or page_num > len(doc):
        return {"count": 0, "mode": "pdf_structure", "panels": [], "error": f"Page {page_num} out of range"}
    
    page = doc[page_num - 1]  # fitz uses 0-indexed
    page_width = page.rect.width
    
    # Get visible image blocks from page structure
    blocks = page.get_text("dict")["blocks"]
    img_blocks = [b for b in blocks if b["type"] == 1]
    text_blocks = [b for b in blocks if b["type"] == 0]
    
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
            skipped.append({"w": round(w, 1), "h": round(h, 1), "x": round(bbox[0], 1), "y": round(bbox[1], 1)})
    
    if skipped:
        print(f"[PDF] Page {page_num}: skipped {len(skipped)} small images: {skipped}", file=sys.stderr, flush=True)
    
    if not raw_panels:
        doc.close()
        return {"count": 0, "mode": "pdf_structure", "panels": []}
    
    # Sort reading order and group by rows
    raw_panels.sort(key=lambda b: b[1])
    row_threshold = 30  # points
    rows = []
    current_row = [raw_panels[0]]
    for p in raw_panels[1:]:
        if abs(p[1] - current_row[0][1]) < row_threshold:
            current_row.append(p)
        else:
            rows.append(sorted(current_row, key=lambda b: b[0]))
            current_row = [p]
    rows.append(sorted(current_row, key=lambda b: b[0]))
    
    # Detect triptychs: a row of 3+ images spanning most of the page width
    # with only 1 shared caption (vs individual captions per image)
    final_panels = []  # list of (bbox, caption_text)
    
    for row in rows:
        row_left = min(b[0] for b in row)
        row_right = max(b[2] for b in row)
        row_bottom = max(b[3] for b in row)
        row_span = row_right - row_left
        
        if len(row) >= 3 and row_span > page_width * 0.85:
            # Could be a triptych — check caption count
            unique_captions = count_captions_for_row(row, row_bottom, text_blocks)
            
            if len(unique_captions) <= 1:
                # One shared caption (or none) -> triptych, merge into single panel
                merged_bbox = (row_left, min(b[1] for b in row), row_right, row_bottom)
                caption_text = extract_caption(unique_captions[0]) if unique_captions else ""
                final_panels.append((merged_bbox, caption_text))
                print(f"[PDF] Page {page_num}: merged {len(row)} images into triptych at y={row[0][1]:.0f}",
                      file=sys.stderr, flush=True)
                continue
        
        # Individual panels — find caption for each
        for bbox in row:
            cap_block = find_caption_for_region(bbox[0], bbox[2], bbox[3], text_blocks)
            caption_text = extract_caption(cap_block) if cap_block else ""
            final_panels.append((bbox, caption_text))
    
    # Render each panel
    mat = fitz.Matrix(zoom, zoom)
    result_panels = []
    
    for bbox, caption_text in final_panels:
        rect = fitz.Rect(bbox)
        pix = page.get_pixmap(matrix=mat, clip=rect)
        
        img_bytes = pix.tobytes("jpeg")
        b64 = base64.b64encode(img_bytes).decode("ascii")
        
        result_panels.append({
            "x": round(bbox[0], 1),
            "y": round(bbox[1], 1),
            "width": round(bbox[2] - bbox[0], 1),
            "height": round(bbox[3] - bbox[1], 1),
            "image": b64,
            "caption": caption_text
        })
    
    doc.close()
    
    print(f"[PDF] Page {page_num}: {len(result_panels)} panels extracted from PDF structure", file=sys.stderr, flush=True)
    
    return {
        "count": len(result_panels),
        "mode": "pdf_structure",
        "panels": result_panels
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
