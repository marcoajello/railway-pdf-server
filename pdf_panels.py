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
      { "x": 15.4, "y": 153.7, "width": 249.6, "height": 107.4, "image": "<base64>" },
      ...
    ]
  }

Panels are sorted in reading order (top-to-bottom, left-to-right).
Small overlay images (icons, stat badges) are filtered by minimum size.
"""
import fitz
import json
import sys
import base64
import io


def extract_panels(pdf_path, page_num, min_w=100, min_h=50, zoom=3):
    """
    Extract panel images from a PDF page using its structural metadata.
    
    1. Read image bounding boxes from PDF structure (get_text dict)
    2. Filter out small overlays by minimum size
    3. Render each panel region at high resolution
    4. Return sorted panels with base64 JPEG data
    """
    doc = fitz.open(pdf_path)
    
    if page_num < 1 or page_num > len(doc):
        return {"count": 0, "mode": "pdf_structure", "panels": [], "error": f"Page {page_num} out of range"}
    
    page = doc[page_num - 1]  # fitz uses 0-indexed
    
    # Get visible image blocks from page structure
    blocks = page.get_text("dict")["blocks"]
    img_blocks = [b for b in blocks if b["type"] == 1]
    
    # Filter: real panels vs small overlays/icons
    panels = []
    skipped = []
    for b in img_blocks:
        bbox = b["bbox"]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w >= min_w and h >= min_h:
            panels.append(bbox)
        else:
            skipped.append({"w": round(w, 1), "h": round(h, 1), "x": round(bbox[0], 1), "y": round(bbox[1], 1)})
    
    if skipped:
        print(f"[PDF] Page {page_num}: skipped {len(skipped)} small images: {skipped}", file=sys.stderr, flush=True)
    
    # Sort reading order: top-to-bottom, left-to-right
    # Group by rows (panels within ~20pt of same y are same row)
    panels.sort(key=lambda b: b[1])  # sort by y first
    
    if panels:
        row_threshold = 20  # points
        rows = []
        current_row = [panels[0]]
        for p in panels[1:]:
            if abs(p[1] - current_row[0][1]) < row_threshold:
                current_row.append(p)
            else:
                rows.append(sorted(current_row, key=lambda b: b[0]))  # sort row by x
                current_row = [p]
        rows.append(sorted(current_row, key=lambda b: b[0]))
        panels = [p for row in rows for p in row]
    
    # Render each panel region and find associated caption text
    mat = fitz.Matrix(zoom, zoom)
    result_panels = []
    
    # Get all text blocks for caption matching
    text_blocks = [b for b in blocks if b["type"] == 0]
    
    for bbox in panels:
        rect = fitz.Rect(bbox)
        pix = page.get_pixmap(matrix=mat, clip=rect)
        
        # Convert to JPEG bytes
        img_bytes = pix.tobytes("jpeg")
        b64 = base64.b64encode(img_bytes).decode("ascii")
        
        # Find caption text below this panel
        # Caption should start near the bottom edge and overlap horizontally
        img_bottom = bbox[3]
        img_left = bbox[0]
        img_right = bbox[2]
        
        caption_block = None
        best_dist = 999
        for tb in text_blocks:
            tb_bbox = tb["bbox"]
            y_dist = tb_bbox[1] - img_bottom
            h_overlap = min(tb_bbox[2], img_right) - max(tb_bbox[0], img_left)
            if -5 <= y_dist < 40 and h_overlap > 30 and y_dist < best_dist:
                caption_block = tb
                best_dist = y_dist
        
        # Extract formatted text from caption
        caption_text = ""
        if caption_block:
            lines = []
            for line in caption_block["lines"]:
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
            caption_text = "\n".join(lines)
        
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
