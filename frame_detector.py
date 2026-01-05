#!/usr/bin/env python3
"""
Rectangle detector for storyboard frames.
Detects frames with black OR gray borders using multiple strategies.
"""
import cv2
import numpy as np
import json
import sys
import base64

def detect_rectangles(image_path):
    """Find rectangles in the image using multiple detection strategies."""
    
    img = cv2.imread(image_path)
    if img is None:
        return []
    
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    all_rectangles = []
    
    # Strategy 1: Black borders (original method)
    _, thresh1 = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    dilated1 = cv2.dilate(thresh1, kernel, iterations=1)
    rects1 = find_rectangles_in_mask(dilated1, width, height)
    all_rectangles.extend(rects1)
    
    # Strategy 2: Gray borders (lighter threshold)
    _, thresh2 = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    dilated2 = cv2.dilate(thresh2, kernel, iterations=2)
    rects2 = find_rectangles_in_mask(dilated2, width, height)
    all_rectangles.extend(rects2)
    
    # Strategy 3: Edge detection (Canny) for any border color
    edges = cv2.Canny(gray, 50, 150)
    dilated3 = cv2.dilate(edges, kernel, iterations=2)
    rects3 = find_rectangles_in_mask(dilated3, width, height)
    all_rectangles.extend(rects3)
    
    # Strategy 4: Adaptive threshold for varying lighting
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
    dilated4 = cv2.dilate(adaptive, kernel, iterations=1)
    rects4 = find_rectangles_in_mask(dilated4, width, height)
    all_rectangles.extend(rects4)
    
    # Deduplicate overlapping rectangles
    rectangles = deduplicate_rectangles(all_rectangles)
    
    # Sort: top to bottom, then left to right (reading order)
    row_threshold = max(150, height // 10)
    rectangles.sort(key=lambda r: (r['y'] // row_threshold, r['x']))
    
    return rectangles

def find_rectangles_in_mask(mask, img_width, img_height):
    """Find valid rectangles in a binary mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rectangles = []
    min_area = (img_width * img_height) * 0.01   # At least 1% of image
    max_area = (img_width * img_height) * 0.5    # At most 50% of image
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter by aspect ratio (storyboard frames are roughly 4:3 to 16:9)
        aspect = w / h if h > 0 else 0
        if aspect < 0.5 or aspect > 2.5:
            continue
        
        # Filter tiny boxes
        if w < 80 or h < 60:
            continue
        
        # Check if it's roughly rectangular (contour area vs bounding box area)
        bbox_area = w * h
        fill_ratio = area / bbox_area if bbox_area > 0 else 0
        if fill_ratio < 0.3:  # Too irregular, not a rectangle
            continue
        
        rectangles.append({
            'x': int(x),
            'y': int(y),
            'width': int(w),
            'height': int(h)
        })
    
    return rectangles

def deduplicate_rectangles(rectangles):
    """Remove overlapping rectangles, keeping the largest one."""
    if not rectangles:
        return []
    
    # Sort by area (largest first)
    rectangles.sort(key=lambda r: r['width'] * r['height'], reverse=True)
    
    kept = []
    for rect in rectangles:
        # Check if this rectangle overlaps significantly with any kept rectangle
        dominated = False
        for kept_rect in kept:
            overlap = compute_overlap(rect, kept_rect)
            if overlap > 0.5:  # More than 50% overlap
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
    
    # Return overlap as fraction of smaller rectangle
    return intersection / min(area1, area2)

def crop_rectangles(image_path, rectangles):
    """Crop and return base64 images for each rectangle."""
    img = cv2.imread(image_path)
    if img is None:
        return []
    
    images = []
    for rect in rectangles:
        x, y, w, h = rect['x'], rect['y'], rect['width'], rect['height']
        
        # Small inset to avoid the border lines
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
        rects = detect_rectangles(image_path)
        result = {'count': len(rects), 'rectangles': rects}
        
        if do_crop:
            result['images'] = crop_rectangles(image_path, rects)
        
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({'error': str(e)}))
        sys.exit(1)
