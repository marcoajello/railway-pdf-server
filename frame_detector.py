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