#!/usr/bin/env python3
"""
Simple rectangle detector for storyboard frames.
Finds black-bordered rectangles. That's it.
"""
import cv2
import numpy as np
import json
import sys
import base64

def detect_rectangles(image_path):
    """Find black rectangles in the image, even if borders are slightly broken."""
    
    img = cv2.imread(image_path)
    if img is None:
        return []
    
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold to get black lines (invert so lines are white)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    
    # Light dilation to connect slightly broken lines - but not enough to bridge to text
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rectangles = []
    min_area = (width * height) * 0.01   # At least 1% of image
    max_area = (width * height) * 0.5    # At most 50% of image
    
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
    
    # Sort: top to bottom, then left to right (reading order)
    # Use adaptive row threshold based on image height (10% of image height)
    row_threshold = max(150, height // 10)
    rectangles.sort(key=lambda r: (r['y'] // row_threshold, r['x']))
    
    return rectangles

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
