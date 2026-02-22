#!/usr/bin/env python3
"""
Rectangle detector for storyboard frames.
Three-path detection:
  1. Bordered — thin dark grid lines → extract panels from grid
  2. Borderless — detect background-colored gaps → infer panels
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
    Tries multiple thresholds to catch colored borders (blue, purple, etc.)
    that would be missed by a single high threshold.
    Returns the result from whichever threshold found the most grid lines.
    """
    best = None
    best_line_count = 0

    for thresh in [180, 140, 100]:
        _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)

        h_kernel_len = max(img_width // 8, 80)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

        v_kernel_len = max(img_height // 8, 80)
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
        v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

        # Guard: solid dark backgrounds produce massive "line" areas
        total_pixels = img_width * img_height
        line_coverage = (np.count_nonzero(h_lines) + np.count_nonzero(v_lines)) / total_pixels
        if line_coverage > 0.15:
            continue

        h_proj = np.sum(h_lines, axis=1) > 0
        h_line_count = np.sum(np.diff(h_proj.astype(int)) == 1)
        v_proj = np.sum(v_lines, axis=0) > 0
        v_line_count = np.sum(np.diff(v_proj.astype(int)) == 1)

        total_lines = h_line_count + v_line_count
        if total_lines > best_line_count:
            best_line_count = total_lines
            best = (h_line_count >= 2 and v_line_count >= 2, h_lines, v_lines)

    if best is None:
        # All thresholds hit the coverage guard — not bordered
        empty = np.zeros((img_height, img_width), dtype=np.uint8)
        return False, empty, empty

    return best


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
        if (w / h if h > 0 else 0) < 0.5 or (w / h if h > 0 else 0) > 4.0:
            continue
        if w < 60 or h < 40:
            continue
        if (area / (w * h) if w * h > 0 else 0) < 0.3:
            continue
        rectangles.append({'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)})

    return rectangles


# ─── Phase 2b: Borderless Gap-Based Detection ──────────────────────────────

def detect_background_color(img, img_width, img_height):
    """
    Detect dominant background color by sampling page edges.
    Works in BGR color space to distinguish pastels from white.
    Returns a 3-element array [B, G, R].
    """
    margin = max(10, min(img_width, img_height) // 20)

    # Sample strips along all four edges
    top = img[0:margin, :].reshape(-1, 3)
    bottom = img[img_height - margin:, :].reshape(-1, 3)
    left = img[:, 0:margin].reshape(-1, 3)
    right = img[:, img_width - margin:].reshape(-1, 3)

    samples = np.vstack([top, bottom, left, right])

    # Median per channel
    bg_color = np.median(samples, axis=0).astype(int)
    return bg_color


def find_gap_runs(profile, threshold=0.7, min_width=8):
    """
    Find runs of background in a 1D profile.
    Returns list of (start, end) tuples.
    """
    is_bg = profile > threshold
    gaps = []
    in_gap = False
    gap_start = 0

    for i in range(len(is_bg)):
        if is_bg[i] and not in_gap:
            gap_start = i
            in_gap = True
        elif not is_bg[i] and in_gap:
            if i - gap_start >= min_width:
                gaps.append((gap_start, i))
            in_gap = False
    if in_gap and len(is_bg) - gap_start >= min_width:
        gaps.append((gap_start, len(is_bg)))

    return gaps


def cluster_gap_positions(all_gap_midpoints, tolerance):
    """
    Cluster gap midpoints from multiple rows.
    Returns list of (position, count) sorted by position.
    """
    if not all_gap_midpoints:
        return []

    sorted_pts = sorted(all_gap_midpoints)
    clusters = []
    cluster = [sorted_pts[0]]

    for pt in sorted_pts[1:]:
        if pt - cluster[-1] <= tolerance:
            cluster.append(pt)
        else:
            clusters.append(cluster)
            cluster = [pt]
    clusters.append(cluster)

    return [(int(np.median(c)), len(c)) for c in clusters]


def extract_borderless_panels(img, gray, img_width, img_height):
    """
    Find panels by detecting background-colored gaps.
    Uses BGR color space so pastels (mint, peach) aren't confused with white.
    1. Find row gaps (horizontal strips of background)
    2. Per-row: find column gaps
    3. For difficult rows: use hints from other rows
    """
    bg_color = detect_background_color(img, img_width, img_height)
    tolerance = 45

    # Background mask: pixel is background only if ALL channels are within tolerance
    diff = np.abs(img.astype(int) - bg_color.astype(int))
    bg_mask = np.all(diff < tolerance, axis=2)  # shape: (H, W)

    h_profile = np.mean(bg_mask, axis=1)
    min_h_gap = max(8, img_height // 80)
    min_content_h = img_height // 15
    min_v_gap = max(4, img_width // 500)
    min_content_w = img_width // 15

    # Step 1: Find row gaps
    h_gaps = find_gap_runs(h_profile, threshold=0.7, min_width=min_h_gap)
    if len(h_gaps) < 2:
        return [], bg_color

    rows = []
    for i in range(len(h_gaps) - 1):
        row_top = h_gaps[i][1]
        row_bottom = h_gaps[i + 1][0]
        if row_bottom - row_top >= min_content_h:
            rows.append((row_top, row_bottom))

    if not rows:
        return [], bg_color

    # Step 2: Per-row column gap detection
    row_gaps = {}       # row_index -> list of (start, end) gap tuples
    row_panels = {}     # row_index -> list of panel rects

    for idx, (row_top, row_bottom) in enumerate(rows):
        row_slice = bg_mask[row_top:row_bottom, :]
        v_profile = np.mean(row_slice, axis=0)
        v_gaps = find_gap_runs(v_profile, threshold=0.7, min_width=min_v_gap)
        row_gaps[idx] = v_gaps

        # Extract panels from this row's gaps
        panels = []
        if len(v_gaps) >= 2:
            for j in range(len(v_gaps) - 1):
                col_left = v_gaps[j][1]
                col_right = v_gaps[j + 1][0]
                if col_right - col_left >= min_content_w:
                    panels.append({
                        'x': int(col_left), 'y': int(row_top),
                        'width': int(col_right - col_left),
                        'height': int(row_bottom - row_top)
                    })
        row_panels[idx] = panels

    # Step 3: Find the best row (most panels found)
    best_row_idx = max(row_panels, key=lambda k: len(row_panels[k]))
    best_count = len(row_panels[best_row_idx])

    if best_count == 0:
        return [], bg_color

    # Collect gap midpoints from all rows that found panels
    all_gap_mids = []
    for idx, gaps in row_gaps.items():
        if len(row_panels[idx]) >= 1:
            for g_start, g_end in gaps:
                mid = (g_start + g_end) // 2
                # Skip edge gaps (page margins)
                if mid > img_width * 0.03 and mid < img_width * 0.97:
                    all_gap_mids.append(mid)

    # Cluster them to find prevailing column boundary positions
    gap_clusters = cluster_gap_positions(all_gap_mids, tolerance=img_width // 20)

    # Step 4: For rows with fewer panels than the best row,
    # re-examine with relaxed threshold near hint positions
    for idx, (row_top, row_bottom) in enumerate(rows):
        if len(row_panels[idx]) >= best_count:
            continue  # this row is fine

        # Get the actual pixel data for this row
        row_gray = gray[row_top:row_bottom, :]
        row_color = img[row_top:row_bottom, :]

        # Try progressively relaxed detection
        found_better = False
        for relax_threshold in [0.5, 0.35, 0.2]:
            # Build a profile using variance — columns with uniform color
            # (any color, not just bg) are more likely to be gaps
            row_var = np.var(row_gray.astype(float), axis=0)
            max_var = np.max(row_var) if np.max(row_var) > 0 else 1
            # Low variance = uniform color = likely gap
            uniformity = 1.0 - (row_var / max_var)

            # Also check bg similarity with relaxed tolerance (in color space)
            wider_diff = np.abs(row_color.astype(int) - bg_color.astype(int))
            wider_bg = np.all(wider_diff < (tolerance * 2), axis=2)
            bg_profile = np.mean(wider_bg, axis=0)

            # Combine: a gap should be either uniform color OR background-like
            combined = np.maximum(uniformity, bg_profile)

            v_gaps_relaxed = find_gap_runs(combined, threshold=relax_threshold, min_width=min_v_gap)

            # Only accept if we find gaps near the hint positions
            if len(v_gaps_relaxed) < 2:
                continue

            panels = []
            for j in range(len(v_gaps_relaxed) - 1):
                col_left = v_gaps_relaxed[j][1]
                col_right = v_gaps_relaxed[j + 1][0]
                if col_right - col_left >= min_content_w:
                    panels.append({
                        'x': int(col_left), 'y': int(row_top),
                        'width': int(col_right - col_left),
                        'height': int(row_bottom - row_top)
                    })

            if len(panels) > len(row_panels[idx]):
                row_panels[idx] = panels
                found_better = True
                break

        # Last resort: use hint positions directly if they align with
        # ANY detectable transition in the row
        if not found_better and len(row_panels[idx]) < 2 and len(gap_clusters) >= 2:
            hint_gaps = []
            for hint_pos, count in gap_clusters:
                if count < 2:
                    continue  # only use well-supported hints
                # Look for ANY column near hint that has above-average uniformity
                search_range = img_width // 25
                left = max(0, hint_pos - search_range)
                right = min(img_width, hint_pos + search_range)

                row_var = np.var(row_gray.astype(float), axis=0)
                region_var = row_var[left:right]
                if len(region_var) == 0:
                    continue

                # Find the most uniform column in the search region
                min_var_offset = np.argmin(region_var)
                best_col = left + int(min_var_offset)
                hint_gaps.append(best_col)

            if len(hint_gaps) >= 2:
                hint_gaps.sort()
                panels = []
                for j in range(len(hint_gaps) - 1):
                    col_left = hint_gaps[j]
                    col_right = hint_gaps[j + 1]
                    if col_right - col_left >= min_content_w:
                        panels.append({
                            'x': int(col_left), 'y': int(row_top),
                            'width': int(col_right - col_left),
                            'height': int(row_bottom - row_top)
                        })
                if len(panels) > len(row_panels[idx]):
                    row_panels[idx] = panels

    # Collect all panels
    rectangles = []
    for idx in sorted(row_panels.keys()):
        rectangles.extend(row_panels[idx])

    return rectangles, bg_color


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
            if compute_overlap(rect, kept_rect) > 0.5:
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
    return intersection / min(r1['width'] * r1['height'], r2['width'] * r2['height'])


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


# ─── Phase 3: Mask-Based Detection (for Vision fallback replacement) ──────

def trim_caption_text(img_gray, rect):
    """
    Trim caption text from the bottom of a detected panel rectangle.
    Scans vertical density profile to find where dense panel content 
    ends and sparse text lines begin.
    """
    x, y, w, h = rect['x'], rect['y'], rect['width'], rect['height']
    
    # Extract the region from the inverted mask (white = content, black = background)
    region = img_gray[y:y+h, x:x+w]
    if region.size == 0 or h < 40:
        return rect
    
    # Calculate row-by-row density (fraction of dark pixels in original = content)
    # img_gray is the original mask: 0=content(dark), 255=background(white)
    # So density = fraction of pixels < 128 per row
    row_density = []
    for row_idx in range(h):
        row = region[row_idx, :]
        dark_fraction = np.sum(row < 128) / w
        row_density.append(dark_fraction)
    
    if not row_density:
        return rect
    
    # Smooth the density signal to avoid noise
    kernel_size = max(3, h // 30)
    if kernel_size % 2 == 0:
        kernel_size += 1
    smoothed = np.convolve(row_density, np.ones(kernel_size) / kernel_size, mode='same')
    
    # Find the main content block: scan from top, find where sustained density drops
    # Text lines have density ~0.05-0.15, panel content has density ~0.3+
    # Look for a sustained drop from the bottom up
    
    # Start from bottom, find first row with substantial content going upward
    content_threshold = 0.08  # below this = mostly empty (gap between text lines or blank)
    text_density_max = 0.20   # text lines are thin, rarely exceed this
    
    # Scan from bottom up to find the text region
    bottom_trim = h  # default: no trim
    
    # Check if bottom 30% has notably lower density than top 70%
    split_point = int(h * 0.7)
    if split_point > 20 and (h - split_point) > 15:
        top_avg = np.mean(smoothed[:split_point])
        bottom_avg = np.mean(smoothed[split_point:])
        
        if top_avg > 0.15 and bottom_avg < top_avg * 0.5:
            # Bottom region is significantly less dense — likely text
            # Find the exact transition: scan downward from split region
            search_start = max(int(h * 0.5), 20)
            gap_count = 0
            for row_idx in range(search_start, h):
                if smoothed[row_idx] < content_threshold:
                    gap_count += 1
                    if gap_count >= 5:  # sustained gap = panel ended
                        bottom_trim = row_idx - gap_count + 1
                        break
                else:
                    gap_count = 0
    
    if bottom_trim < h and bottom_trim > h * 0.4:
        return {
            'x': rect['x'], 'y': rect['y'],
            'width': rect['width'], 'height': bottom_trim,
            'area': rect.get('area', rect['width'] * bottom_trim),
            'fill': rect.get('fill', 0.5)
        }
    
    return rect


def detect_from_mask(mask_path, expected_count=0):
    """
    Detect panels from a pre-thresholded binary mask image.
    The mask has black content on white background (as generated by sharp.threshold()).
    
    Uses two strategies:
    1. Light morphology — works for solid content (drawings, ink)
    2. Aggressive morphology — works for photo panels (fill holes from varied brightness)
    
    After detection, trims caption text from bottom of each panel.
    """
    img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []

    height, width = img.shape[:2]
    img_area = width * height
    
    # Invert: we need white content on black background for findContours
    inverted = cv2.bitwise_not(img)
    
    candidates = None
    
    # Strategy 1: Light morphology (good for solid drawings/ink)
    h_close = np.ones((1, 15), np.uint8)
    v_close = np.ones((3, 1), np.uint8)
    closed1 = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, h_close, iterations=2)
    closed1 = cv2.morphologyEx(closed1, cv2.MORPH_CLOSE, v_close, iterations=1)
    
    candidates = find_panel_contours(closed1, width, height, img_area, expected_count)
    
    print(f"[Mask] Strategy 1 (light): found {len(candidates)} candidates", file=sys.stderr, flush=True)
    
    if candidates and len(candidates) >= max(1, int(expected_count * 0.7)):
        # Strategy 1 worked — trim text and return
        candidates = [trim_caption_text(img, c) for c in candidates]
        candidates = finalize_candidates(candidates, expected_count, height)
        return candidates
    
    # Strategy 2: Aggressive morphology (fill holes in photos)
    # Large close kernel fills white holes inside photos from varied brightness
    big_close = np.ones((25, 25), np.uint8)
    closed2 = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, big_close, iterations=3)
    
    # Then erode back to separate panels that may have merged
    # Use a thin vertical kernel to re-open horizontal gaps between columns
    h_erode = np.ones((1, 15), np.uint8)
    closed2 = cv2.morphologyEx(closed2, cv2.MORPH_OPEN, h_erode, iterations=1)
    # And a thin horizontal kernel to re-open vertical gaps between rows  
    v_erode = np.ones((10, 1), np.uint8)
    closed2 = cv2.morphologyEx(closed2, cv2.MORPH_OPEN, v_erode, iterations=1)
    
    candidates2 = find_panel_contours(closed2, width, height, img_area, expected_count)
    
    print(f"[Mask] Strategy 2 (aggressive): found {len(candidates2)} candidates", file=sys.stderr, flush=True)
    
    if candidates2 and len(candidates2) >= max(1, int(expected_count * 0.7)):
        candidates2 = [trim_caption_text(img, c) for c in candidates2]
        candidates2 = finalize_candidates(candidates2, expected_count, height)
        return candidates2
    
    # Strategy 3: Even more aggressive — dilate first, then close
    dilated = cv2.dilate(inverted, np.ones((15, 15), np.uint8), iterations=3)
    closed3 = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, np.ones((30, 30), np.uint8), iterations=2)
    # Erode back to approximate original panel sizes
    closed3 = cv2.erode(closed3, np.ones((15, 15), np.uint8), iterations=3)
    # Open to separate merged panels
    closed3 = cv2.morphologyEx(closed3, cv2.MORPH_OPEN, np.ones((5, 20), np.uint8), iterations=1)
    closed3 = cv2.morphologyEx(closed3, cv2.MORPH_OPEN, np.ones((15, 5), np.uint8), iterations=1)
    
    candidates3 = find_panel_contours(closed3, width, height, img_area, expected_count)
    
    print(f"[Mask] Strategy 3 (dilate+close): found {len(candidates3)} candidates", file=sys.stderr, flush=True)
    
    if candidates3 and len(candidates3) >= max(1, int(expected_count * 0.7)):
        candidates3 = [trim_caption_text(img, c) for c in candidates3]
        candidates3 = finalize_candidates(candidates3, expected_count, height)
        return candidates3
    
    # Nothing worked
    return []


def find_panel_contours(binary_img, width, height, img_area, expected_count):
    """Extract panel-sized rectangular contours from a binary image."""
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    min_area = img_area * 0.012
    max_area = img_area * 0.45
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        
        aspect = w / h if h > 0 else 0
        if aspect < 0.35 or aspect > 4.0:
            continue
        
        if w < 50 or h < 30:
            continue
        
        bbox_area = w * h
        fill_ratio = area / bbox_area if bbox_area > 0 else 0
        if fill_ratio < 0.3:
            continue
        
        candidates.append({
            'x': int(x), 'y': int(y),
            'width': int(w), 'height': int(h),
            'area': area, 'fill': fill_ratio
        })
    
    return candidates


def finalize_candidates(candidates, expected_count, height):
    """Deduplicate, sort by area, trim to expected count, sort reading order."""
    if not candidates:
        return []
    
    candidates = deduplicate_rectangles(candidates)
    candidates.sort(key=lambda r: r['area'], reverse=True)
    
    if expected_count > 0 and len(candidates) > expected_count:
        candidates = candidates[:expected_count]
    
    for c in candidates:
        c.pop('area', None)
        c.pop('fill', None)
    
    row_threshold = max(150, height // 10)
    candidates.sort(key=lambda r: (r['y'] // row_threshold, r['x']))
    
    return candidates


# ─── Main Entry Point ──────────────────────────────────────────────────────

def detect_rectangles(image_path):
    """
    Three-path detection:
      mode: grid    → bordered panels found via grid lines
      mode: content → borderless panels found via gap detection
      mode: vision  → neither worked, caller should use Vision API
    """
    img = cv2.imread(image_path)
    if img is None:
        return [], False, 'error'

    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Phase 1: Border gate
    is_bordered, h_lines, v_lines = detect_borders(gray, width, height)

    if is_bordered:
        rectangles = extract_bordered_panels(gray, h_lines, v_lines, width, height)
        rectangles = deduplicate_rectangles(rectangles)
        row_threshold = max(150, height // 10)
        rectangles.sort(key=lambda r: (r['y'] // row_threshold, r['x']))
        return rectangles, True, 'grid'

    # Phase 2: Borderless — gap-based detection
    content_rects, bg_color = extract_borderless_panels(img, gray, width, height)

    if len(content_rects) >= 2:
        row_threshold = max(150, height // 10)
        content_rects.sort(key=lambda r: (r['y'] // row_threshold, r['x']))
        return content_rects, False, 'content'

    # Phase 3: Neither worked
    return [], False, 'vision'


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(json.dumps({'error': 'No image path'}))
        sys.exit(1)

    image_path = sys.argv[1]
    mode_arg = sys.argv[2] if len(sys.argv) > 2 else 'crop'
    
    try:
        if mode_arg == 'mask':
            # Mask mode: detect panels from pre-thresholded mask image
            # argv[3] = expected count (optional)
            # Returns rectangles only — Node handles full-res cropping
            expected = int(sys.argv[3]) if len(sys.argv) > 3 else 0
            
            rects = detect_from_mask(image_path, expected)
            result = {
                'count': len(rects),
                'mode': 'mask',
                'rectangles': rects
            }
            print(json.dumps(result))
        else:
            # Standard mode: detect from original image
            do_crop = mode_arg == 'crop'
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