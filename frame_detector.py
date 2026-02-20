#!/usr/bin/env python3
"""
Rectangle detector for storyboard frames.
Two-phase approach:
  Phase 1: Border detection gate — are there actual panel borders?
  Phase 2: If bordered, extract panels via grid line morphology.
           If borderless, return immediately (let Vision handle it).
"""
import cv2
import numpy as np
import json
import sys
import base64

def detect_borders(gray, img_width, img_height):
    """
    Determine whether this page has bordered panels.
    Uses morphological line detection to find long horizontal and vertical lines.
    Returns (is_bordered, h_lines_mask, v_lines_mask) so we can reuse the masks.
    """
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    
    # Horizontal lines: kernel must be long enough to span a panel width
    h_kernel_len = max(img_width // 8, 80)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    
    # Vertical lines: kernel must be tall enough to span a panel height
    v_kernel_len = max(img_height // 8, 80)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    
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
    
    return is_bordered, h_lines, v_lines, h_line_count, v_line_count

def extract_bordered_panels(gray, h_lines, v_lines, img_width, img_height):
    """
    Extract panel rectangles from detected grid lines.
    Reuses the h/v line masks from the border detection phase.
    """
    # Combine horizontal and vertical lines
    grid = cv2.add(h_lines, v_lines)
    
    # Dilate to close small gaps in lines
    dilate_kernel = np.ones((3, 3), np.uint8)
    grid = cv2.dilate(grid, dilate_kernel, iterations=2)
    
    # Find contours in the grid
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
        
        rectangles.append({
            'x': int(x),
            'y': int(y),
            'width': int(w),
            'height': int(h)
        })
    
    return rectangles

def detect_rectangles(image_path):
    """Main entry point. Runs border gate, then appropriate extraction."""
    
    img = cv2.imread(image_path)
    if img is None:
        return [], False
    
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Phase 1: Border detection gate
    is_bordered, h_lines, v_lines, h_count, v_count = detect_borders(gray, width, height)
    
    if not is_bordered:
        # Borderless — don't waste time, let Vision handle it
        return [], False
    
    # Phase 2: Extract panels from grid lines
    rectangles = extract_bordered_panels(gray, h_lines, v_lines, width, height)
    rectangles = deduplicate_rectangles(rectangles)
    
    # Sort: top to bottom, then left to right (reading order)
    row_threshold = max(150, height // 10)
    rectangles.sort(key=lambda r: (r['y'] // row_threshold, r['x']))
    
    return rectangles, True

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
        cropped = img[y+inset:y+h-inset, x+inset:x+w-inset]
        
        if cropped.size == 0:
            images.append(None)
            continue
        
        _, buffer = cv2.imencode('.jpg', cropped, [cv2.IMWRITE_JPEG_QUALITY, 85])
        images.append(base64.b64encode(buffer).decode('utf-8'))
    
    return images

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(json.dumps({'error': 'No image path'}))
        sys.exit(1)
    
    image_path = sys.argv[1]
    do_crop = len(sys.argv) > 2 and sys.argv[2] == 'crop'
    
    try:
        rects, bordered = detect_rectangles(image_path)
        result = {
            'count': len(rects),
            'bordered': bordered,
            'rectangles': rects
        }
        
        if do_crop:
            result['images'] = crop_rectangles(image_path, rects)
        
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({'error': str(e)}))
        sys.exit(1)
