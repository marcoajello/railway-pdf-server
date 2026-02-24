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
      { "x": 15.4, "y": 153.7, "width": 249.6, "height": 107.4,
        "image": "<base64>", "images": ["<base64>"], "caption": "..." },
      ...
    ]
  }

For triptych/pan sequences, "image" is the first sub-image and "images" contains
all individual sub-images so the client can display them side by side.
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


def find_caption_for_region(img_left, img_right, img_bottom, text_blocks, max_y_dist=80):
    """Find the caption text block below an image region.
    Also looks for frame number labels just above the image."""
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


def find_all_captions_for_region(img_left, img_right, img_top, img_bottom, text_blocks, max_y_dist=80):
    """Find ALL caption text blocks near an image region (below, above, or beside).
    Returns list of (text_block, position, distance) tuples."""
    results = []
    for tb in text_blocks:
        tb_bbox = tb["bbox"]
        h_overlap = min(tb_bbox[2], img_right) - max(tb_bbox[0], img_left)

        # Text below image (caption)
        if h_overlap > 20:
            y_dist_below = tb_bbox[1] - img_bottom
            if -5 <= y_dist_below < max_y_dist:
                results.append((tb, 'below', y_dist_below))

        # Text above or at top of image (frame number labels)
        # These can be small text blocks that don't overlap much horizontally
        v_overlap = min(tb_bbox[3], img_top + 60) - max(tb_bbox[1], img_top - 60)
        if v_overlap > 0 or (img_top - tb_bbox[3] >= -5 and img_top - tb_bbox[3] < 60):
            # Check if this is a small text block (likely a frame number)
            tb_text = "".join(s["text"] for line in tb["lines"] for s in line["spans"]).strip()
            if len(tb_text) <= 10:
                # Small text near top of image — likely a frame number
                y_dist_above = abs(tb_bbox[1] - img_top)
                # Must be horizontally near the image
                if tb_bbox[0] < img_right + 20 and tb_bbox[2] > img_left - 20:
                    results.append((tb, 'above', y_dist_above))

    return results


def extract_frame_numbers_for_row(row_panels, row_top, text_blocks):
    """
    Find frame number labels for a row of panels.
    Handles cases like "1.2 | 1.3" in a single text block spanning multiple panels,
    or individual "1.1" labels above each panel.

    Uses span-level bounding boxes for precise X-position matching.
    """
    import re
    frame_numbers = ['' for _ in row_panels]
    number_pattern = re.compile(r'^\d+(?:\.\d+)?[A-Za-z]?$')

    for tb in text_blocks:
        tb_bbox = tb["bbox"]
        # Look for text blocks near the top of this row (above or at start)
        y_dist = row_top - tb_bbox[3]  # distance from bottom of text to top of row
        if not (-10 <= y_dist < 60):
            # Also check if text block is at the same Y as row start
            if not (abs(tb_bbox[1] - row_top) < 60):
                continue

        # Process at span level for precise X-position matching
        # Each span has its own bounding box, so "1.2" at x=12 and "1.3" at x=977
        # will correctly match to different panels
        for line in tb["lines"]:
            for span in line["spans"]:
                span_text = span["text"].strip()
                if not span_text or not number_pattern.match(span_text):
                    continue

                # Match this number to the closest panel by X position
                span_center_x = (span["bbox"][0] + span["bbox"][2]) / 2
                best_idx = -1
                best_dist = 999999
                for i, bbox in enumerate(row_panels):
                    panel_center_x = (bbox[0] + bbox[2]) / 2
                    dist = abs(span_center_x - panel_center_x)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = i
                if best_idx >= 0 and not frame_numbers[best_idx]:
                    frame_numbers[best_idx] = span_text

    return frame_numbers


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


def render_panel(page, bbox, mat):
    """Render a single panel region to base64 JPEG."""
    rect = fitz.Rect(bbox)
    pix = page.get_pixmap(matrix=mat, clip=rect)
    img_bytes = pix.tobytes("jpeg")
    return base64.b64encode(img_bytes).decode("ascii")


def extract_panels(pdf_path, page_num, min_w=100, min_h=50, zoom=3):
    """
    Extract panel images from a PDF page using its structural metadata.
    """
    doc = fitz.open(pdf_path)
    
    if page_num < 1 or page_num > len(doc):
        return {"count": 0, "mode": "pdf_structure", "panels": [], "error": f"Page {page_num} out of range"}
    
    page = doc[page_num - 1]
    page_width = page.rect.width
    
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
    
    # Sort and group by rows
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
    
    # Detect triptychs and build final panel list
    mat = fitz.Matrix(zoom, zoom)
    result_panels = []
    
    for row in rows:
        row_left = min(b[0] for b in row)
        row_right = max(b[2] for b in row)
        row_top = min(b[1] for b in row)
        row_bottom = max(b[3] for b in row)
        row_span = row_right - row_left

        # Extract frame numbers for this row of panels
        row_frame_numbers = extract_frame_numbers_for_row(row, row_top, text_blocks)

        if len(row) >= 2 and row_span > page_width * 0.60:
            # Could be a multi-image frame (diptych, triptych, or more)
            # Key signal: multiple images share ONE caption below them
            unique_captions = count_captions_for_row(row, row_bottom, text_blocks)

            if len(unique_captions) <= 1 and len(unique_captions) < len(row):
                # Multi-image frame: one logical frame with multiple sub-images
                sub_images = [render_panel(page, bbox, mat) for bbox in row]
                caption_text = extract_caption(unique_captions[0]) if unique_captions else ""

                # Use the first frame number from the row (triptych = single logical frame)
                triptych_frame_num = row_frame_numbers[0] if row_frame_numbers else ""

                # Use first sub-image as primary, include all in images array
                result_panels.append({
                    "x": round(row_left, 1),
                    "y": round(row_top, 1),
                    "width": round(row_span, 1),
                    "height": round(row_bottom - row_top, 1),
                    "image": sub_images[0],
                    "images": sub_images,
                    "caption": caption_text,
                    "frameLabel": triptych_frame_num,
                    "triptych": True
                })
                print(f"[PDF] Page {page_num}: triptych at y={row[0][1]:.0f} ({len(row)} sub-images)",
                      file=sys.stderr, flush=True)
                continue
        
        # Individual panels
        for panel_idx, bbox in enumerate(row):
            b64 = render_panel(page, bbox, mat)

            # Find caption below
            cap_block = find_caption_for_region(bbox[0], bbox[2], bbox[3], text_blocks)
            caption_text = extract_caption(cap_block) if cap_block else ""

            # Use the frame number extracted for this panel's position in the row
            frame_label = row_frame_numbers[panel_idx] if panel_idx < len(row_frame_numbers) else ""

            result_panels.append({
                "x": round(bbox[0], 1),
                "y": round(bbox[1], 1),
                "width": round(bbox[2] - bbox[0], 1),
                "height": round(bbox[3] - bbox[1], 1),
                "image": b64,
                "images": [b64],
                "caption": caption_text,
                "frameLabel": frame_label
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
