#!/usr/bin/env python3
"""
Storyboard frame detector using OpenCV
Finds rectangular frames in storyboard images
"""

import cv2
import numpy as np
import json
import sys
import base64
from io import BytesIO

def detect_frames(image_data, min_area_ratio=0.01, max_area_ratio=0.4):
    """
    Detect rectangular frames in a storyboard image.
    
    Args:
        image_data: Base64 encoded image or file path
        min_area_ratio: Minimum frame area as ratio of image area
        max_area_ratio: Maximum frame area as ratio of image area
    
    Returns:
        List of bounding boxes [x, y, width, height] in pixels
    """
    
    # Load image
    if image_data.startswith('/'):
        # File path
        img = cv2.imread(image_data)
    else:
        # Base64
        img_bytes = base64.b64decode(image_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if img is None:
        return []
    
    height, width = img.shape[:2]
    img_area = width * height
    min_area = img_area * min_area_ratio
    max_area = img_area * max_area_ratio
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive threshold to handle varying lighting
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Also try regular threshold for cleaner lines
    _, thresh2 = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Combine thresholds
    combined = cv2.bitwise_or(thresh, thresh2)
    
    # Morphological operations to clean up
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate edges to connect broken lines
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    frames = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter by area
        if area < min_area or area > max_area:
            continue
        
        # Approximate contour to polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter by aspect ratio (frames are usually somewhat rectangular)
        aspect = w / h if h > 0 else 0
        if aspect < 0.3 or aspect > 3.0:
            continue
        
        # Filter out very thin rectangles (likely borders/lines)
        if w < 50 or h < 50:
            continue
        
        # Check if it's roughly rectangular (4 corners)
        if len(approx) >= 4 and len(approx) <= 8:
            frames.append({
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h),
                'area': int(area)
            })
    
    # If we didn't find good frames, try a grid-based approach
    if len(frames) < 2:
        frames = detect_grid_frames(img, gray)
    
    # Sort by position (top to bottom, left to right)
    frames.sort(key=lambda f: (f['y'] // 100, f['x']))
    
    # Remove duplicates/overlapping frames
    frames = remove_overlapping(frames)
    
    return frames


def detect_grid_frames(img, gray):
    """
    Detect frames using line detection for grid-based storyboards.
    """
    height, width = img.shape[:2]
    frames = []
    
    # Detect lines using HoughLinesP
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    
    if lines is None:
        return frames
    
    # Separate horizontal and vertical lines
    h_lines = []
    v_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        
        if angle < 10 or angle > 170:  # Horizontal
            h_lines.append((min(y1, y2), max(y1, y2)))
        elif 80 < angle < 100:  # Vertical
            v_lines.append((min(x1, x2), max(x1, x2)))
    
    # Cluster lines to find grid
    h_positions = cluster_lines([l[0] for l in h_lines], threshold=30)
    v_positions = cluster_lines([l[0] for l in v_lines], threshold=30)
    
    # Add image boundaries
    if 0 not in h_positions:
        h_positions = [0] + h_positions
    if height not in h_positions:
        h_positions.append(height)
    if 0 not in v_positions:
        v_positions = [0] + v_positions
    if width not in v_positions:
        v_positions.append(width)
    
    h_positions.sort()
    v_positions.sort()
    
    # Create grid cells
    min_cell_size = min(width, height) * 0.1
    
    for i in range(len(h_positions) - 1):
        for j in range(len(v_positions) - 1):
            y1, y2 = h_positions[i], h_positions[i + 1]
            x1, x2 = v_positions[j], v_positions[j + 1]
            w, h = x2 - x1, y2 - y1
            
            if w > min_cell_size and h > min_cell_size:
                frames.append({
                    'x': int(x1),
                    'y': int(y1),
                    'width': int(w),
                    'height': int(h),
                    'area': int(w * h)
                })
    
    return frames


def cluster_lines(positions, threshold=30):
    """Cluster nearby line positions."""
    if not positions:
        return []
    
    positions = sorted(set(positions))
    clusters = [[positions[0]]]
    
    for pos in positions[1:]:
        if pos - clusters[-1][-1] < threshold:
            clusters[-1].append(pos)
        else:
            clusters.append([pos])
    
    # Return average of each cluster
    return [int(sum(c) / len(c)) for c in clusters]


def remove_overlapping(frames, overlap_threshold=0.5):
    """Remove frames that overlap significantly."""
    if not frames:
        return frames
    
    # Sort by area (keep larger frames)
    frames.sort(key=lambda f: f['area'], reverse=True)
    
    keep = []
    for frame in frames:
        is_overlapping = False
        for kept in keep:
            # Calculate overlap
            x1 = max(frame['x'], kept['x'])
            y1 = max(frame['y'], kept['y'])
            x2 = min(frame['x'] + frame['width'], kept['x'] + kept['width'])
            y2 = min(frame['y'] + frame['height'], kept['y'] + kept['height'])
            
            if x1 < x2 and y1 < y2:
                overlap_area = (x2 - x1) * (y2 - y1)
                min_area = min(frame['area'], kept['area'])
                if overlap_area / min_area > overlap_threshold:
                    is_overlapping = True
                    break
        
        if not is_overlapping:
            keep.append(frame)
    
    return keep


def crop_frame(image_path, bbox, output_path=None):
    """
    Crop a frame from an image.
    
    Args:
        image_path: Path to source image
        bbox: Dict with x, y, width, height
        output_path: Optional path to save cropped image
    
    Returns:
        Base64 encoded cropped image
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
    cropped = img[y:y+h, x:x+w]
    
    if output_path:
        cv2.imwrite(output_path, cropped)
    
    # Encode to base64
    _, buffer = cv2.imencode('.jpg', cropped, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode('utf-8')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(json.dumps({'error': 'No image path provided'}))
        sys.exit(1)
    
    image_path = sys.argv[1]
    crop_output = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        frames = detect_frames(image_path)
        
        result = {
            'frames': frames,
            'count': len(frames)
        }
        
        # If crop_output is 'crop', also return cropped images
        if crop_output == 'crop':
            cropped_images = []
            for i, frame in enumerate(frames):
                cropped = crop_frame(image_path, frame)
                if cropped:
                    cropped_images.append(cropped)
            result['images'] = cropped_images
        
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({'error': str(e)}))
        sys.exit(1)
