#!/usr/bin/env python3
"""
Rectangle detector for storyboard frames.
Three-path detection:
  1. Bordered — thin grid lines detected → extract panels from grid
  2. Borderless — uniform background detected → find content regions
  3. Unknown — neither works → signal Vision fallback
"""
import cv2
import numpy as np
import json
import sys
import base64


# ─── Phase 1: Border Detection Gate ────────────────────────────────────────

def detect_borders(gray, img_width, img_height):
    """
    Determine whether this page has bordered panels.
    Returns (is_bordered, h_lines_mask, v_lines_mask).
    Guards against solid dark backgrounds by checking line thickness.
    """
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # Horizontal lines
    h_kernel_len = max(img_width // 8, 80)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    # Vertical lines
    v_kernel_len = max(img_height // 8, 80)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    # Guard: if line masks cover too much area, it's a solid background, not borders
    total_pixels = img_width * img_height
    line_coverage = (np.count_nonzero(h_lines) + np.count_nonzero(v_lines)) / total_pixels
    if line_coverage > 0.15:
        return False, h_lines, v_lines

    # Count distinct horizontal lines by projecting onto Y axis
    h_proj = np.sum(h_lines, axis=1) > 0
    h_runs = np.diff(h_proj.astype(int))
    h_line_count = np.sum(h_runs == 1)

    # Count distinct vertical lines by projecting onto X axis
    v_proj = np.sum(v_lines, axis=0) > 0
    v_runs = np.diff(v_proj.astype(int))
    v_line_count = np.sum(v_runs == 1)

    # Need at least 2 horizontal AND 2 vertical lines to form panel borders
    is_bordered = h_line_count >= 2 and v_line_count >= 2

    return is_bordered, h_lines, v_lines


# ─── Phase 2a: Bordered Panel Extraction ───────────────────────────────────

def extract_bordered_panels(gray, h_lines, v_lines, img_width, img_height):
    """Extract panel rectangles from detected grid lines."""
    grid = cv2.add(h_lines, v_lines)

    dilate_kernel = np.ones((3, 3), np.uint8)
    grid = cv2.dilate(grid, dilate_kernel, iterations=2)

    contours, _ = cv2.findContours(grid, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    rectangles = []
    min_area = (img_width * img_height) * 0.008
    max_area = (img_width * img_height) * 0.45

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        aspect = w / h if h > 0 else 0
        if aspect < 0.5 or aspect > 4.0:
            continue

        if w < 60 or h < 40:
            continue

        bbox_area = w * h
        fill_ratio = area / bbox_area if bbox_area > 0 else 0
        if fill_ratio < 0.3:
            continue

        rectangles.append({'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)})

    return rectangles


# ─── Phase 2b: Borderless Content Region Detection ─────────────────────────

def detect_background_color(gray, img_width, img_height):
    """
    Detect the dominant background color by sampling page edges and corners.
    Returns the median grayscale value of the background.
    """
    samples = []
    margin = max(10, min(img_width, img_height) // 20)

    # Sample strips along all four edges
    samples.append(gray[0:margin, :].flatten())                 # top
    samples.append(gray[img_height - margin:, :].flatten())     # bottom
    samples.append(gray[:, 0:margin].flatten())                 # left
    samples.append(gray[:, img_width - margin:].flatten())      # right

    all_samples = np.concatenate(samples)
    bg_value = int(np.median(all_samples))

    return bg_value


def extract_borderless_panels(gray, img_width, img_height):
    """
    Find content regions (photos/drawings) on a uniform background.
    Works for white, black, or solid-color backgrounds.
    """
    bg_value = detect_background_color(gray, img_width, img_height)

    # Threshold: find pixels that differ significantly from background
    diff_threshold = 30

    if bg_value > 128:
        # Light background — content is darker
        _, content_mask = cv2.threshold(gray, bg_value - diff_threshold, 255, cv2.THRESH_BINARY_INV)
    else:
        # Dark background — content is lighter
        _, content_mask = cv2.threshold(gray, bg_value + diff_threshold, 255, cv2.THRESH_BINARY)

    # Close small gaps within individual panels (fill holes in photos)
    close_kernel = np.ones((5, 5), np.uint8)
    content_mask = cv2.morphologyEx(content_mask, cv2.MORPH_CLOSE, close_kernel, iterations=3)

    # Gentle dilate to solidify panel regions without merging neighbors
    dilate_kernel = np.ones((5, 5), np.uint8)
    content_mask = cv2.dilate(content_mask, dilate_kernel, iterations=2)

    # Erode to separate text captions from images
    # Text is thin so it gets eroded away; photos are large and survive
    erode_kernel = np.ones((8, 8), np.uint8)
    content_mask = cv2.erode(content_mask, erode_kernel, iterations=2)

    # Restore panel edges slightly after erosion
    restore_kernel = np.ones((4, 4), np.uint8)
    content_mask = cv2.dilate(content_mask, restore_kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(content_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rectangles = []
    min_area = (img_width * img_height) * 0.005  # Lower threshold for borderless
    max_area = (img_width * img_height) * 0.5

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        aspect = w / h if h > 0 else 0
        if aspect < 0.3 or aspect > 5.0:
            continue

        if w < 50 or h < 30:
            continue

        rectangles.append({'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)})

    return rectangles, bg_value


# ─── Shared Utilities ──────────────────────────────────────────────────────

def deduplicate_rectangles(rectangles):
    """Remove overlapping rectangles, keeping the largest one."""
    if not rectangles:
        return []

    rectangles.sort(key=lambda r: r['width'] * r['height'], reverse=True)

    kept = []
    for rect in rectangles:
        dominated = False
        for kept_rect in kept:
            overlap = compute_overlap(rect, kept_rect)
            if overlap > 0.5:
                dominated = True
                break
        if not dominated:
            kept.append(rect)

    return kept


def compute_overlap(r1, r2):
    """Compute overlap ratio between two rectangles."""
    x1 = max(r1['x'], r2['x'])
    y1 = max(r1['y'], r2['y'])
    x2 = min(r1['x'] + r1['width'], r2['x'] + r2['width'])
    y2 = min(r1['y'] + r1['height'], r2['y'] + r2['height'])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = r1['width'] * r1['height']
    area2 = r2['width'] * r2['height']

    return intersection / min(area1, area2)


def crop_rectangles(image_path, rectangles):
    """Crop and return base64 images for each rectangle."""
    img = cv2.imread(image_path)
    if img is None:
        return []

    images = []
    for rect in rectangles:
        x, y, w, h = rect['x'], rect['y'], rect['width'], rect['height']

        inset = 3
        cropped = img[y + inset:y + h - inset, x + inset:x + w - inset]

        if cropped.size == 0:
            images.append(None)
            continue

        _, buffer = cv2.imencode('.jpg', cropped, [cv2.IMWRITE_JPEG_QUALITY, 85])
        images.append(base64.b64encode(buffer).decode('utf-8'))

    return images


# ─── Main Entry Point ──────────────────────────────────────────────────────

def detect_rectangles(image_path):
    """
    Main entry point. Three-path detection:
      1. bordered: true  → OpenCV grid extraction
      2. bordered: false, mode: content → deterministic content region detection
      3. bordered: false, mode: vision  → signal caller to use Vision API
    """
    img = cv2.imread(image_path)
    if img is None:
        return [], False, 'error'

    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Phase 1: Border gate
    is_bordered, h_lines, v_lines = detect_borders(gray, width, height)[:3]

    if is_bordered:
        rectangles = extract_bordered_panels(gray, h_lines, v_lines, width, height)
        rectangles = deduplicate_rectangles(rectangles)
        row_threshold = max(150, height // 10)
        rectangles.sort(key=lambda r: (r['y'] // row_threshold, r['x']))
        return rectangles, True, 'grid'

    # Phase 2: Borderless — try content region detection
    content_rects, bg_value = extract_borderless_panels(gray, width, height)
    content_rects = deduplicate_rectangles(content_rects)

    if len(content_rects) >= 2:
        # Found enough content regions — use them
        row_threshold = max(150, height // 10)
        content_rects.sort(key=lambda r: (r['y'] // row_threshold, r['x']))
        return content_rects, False, 'content'

    # Phase 3: Neither approach worked — signal Vision fallback
    return [], False, 'vision'


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(json.dumps({'error': 'No image path'}))
        sys.exit(1)

    image_path = sys.argv[1]
    do_crop = len(sys.argv) > 2 and sys.argv[2] == 'crop'

    try:
        rects, bordered, mode = detect_rectangles(image_path)
        result = {
            'count': len(rects),
            'bordered': bordered,
            'mode': mode,
            'rectangles': rects
        }

        if do_crop:
            result['images'] = crop_rectangles(image_path, rects)

        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({'error': str(e)}))
        sys.exit(1)
