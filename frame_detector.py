#!/usr/bin/env python3
"""
Simple rectangle detector for storyboard frames.
Finds black-bordered rectangles - the actual illustration frames, not outer containers.
"""
import cv2
import numpy as np
import json
import sys
import base64

def detect_rectangles(image_path):
    """Find black rectangles in the image, preferring inner frames over containers."""
    
    img = cv2.imread(image_path)
    if img is None:
        return []
    
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold to get black lines (invert so lines are white)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    
    # Dilate to connect broken lines - key for imperfect rectangles
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    
    # Find ALL contours with hierarchy (not just external)
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if hierarchy is None:
        return []
    
    hierarchy = hierarchy[0]  # Unwrap
    
    rectangles = []
    min_area = (width * height) * 0.01   # At least 1% of image
    max_area = (width * height) * 0.5    # At most 50% of image
    
    candidates = []
    
    for i, contour in enumerate(contours):
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
        
        # Check hierarchy: [next, prev, first_child, parent]
        has_parent = hierarchy[i][3] != -1
        has_child = hierarchy[i][2] != -1
        
        candidates.append({
            'x': int(x),
            'y': int(y),
            'width': int(w),
            'height': int(h),
            'area': area,
            'has_parent': has_parent,
            'has_child': has_child,
            'index': i
        })
    
    # Filter: if a rectangle contains another rectangle, check if outer is a container
    # (has whitespace/text below inner) vs parallax (artwork fills the space)
    final_rects = []
    
    for cand in candidates:
        dominated = False
        for other in candidates:
            if other['index'] == cand['index']:
                continue
            
            # Check if cand contains other (other is inside cand)
            if (other['x'] >= cand['x'] and 
                other['y'] >= cand['y'] and
                other['x'] + other['width'] <= cand['x'] + cand['width'] and
                other['y'] + other['height'] <= cand['y'] + cand['height']):
                
                # Inner must be substantial (not noise)
                if other['area'] < cand['area'] * 0.3:
                    continue
                
                # Check the gap below the inner rectangle
                # If there's significant space below, outer is likely a container with text
                gap_below = (cand['y'] + cand['height']) - (other['y'] + other['height'])
                gap_ratio = gap_below / cand['height'] if cand['height'] > 0 else 0
                
                # Also check if inner frame is near the top of outer (container pattern)
                gap_above = other['y'] - cand['y']
                gap_above_ratio = gap_above / cand['height'] if cand['height'] > 0 else 0
                
                # Container pattern: inner near top, significant gap below (for text)
                # Parallax pattern: inner fills most of outer, gaps are similar on all sides
                if gap_ratio > 0.1 and gap_above_ratio < 0.05:
                    # Looks like container - frame at top, text area below
                    dominated = True
                    break
        
        if not dominated:
            final_rects.append({
                'x': cand['x'],
                'y': cand['y'],
                'width': cand['width'],
                'height': cand['height']
            })
    
    # Sort: top to bottom, then left to right (reading order)
    final_rects.sort(key=lambda r: (r['y'] // 80, r['x']))
    
    return final_rects


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
