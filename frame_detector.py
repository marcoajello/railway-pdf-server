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
    Uses whitespace-band detection: scans bottom-up for a full-width horizontal
    strip of near-zero dark pixel density (the gap between drawing and caption).
    This works regardless of whether the panel content is dense (photos) or
    sparse (hand-drawn lines) because the gap is a structural feature of
    every storyboard layout.
    """
    x, y, w, h = rect['x'], rect['y'], rect['width'], rect['height']

    # img_gray is the original mask: 0=content(dark), 255=background(white)
    region = img_gray[y:y+h, x:x+w]
    if region.size == 0 or h < 60:
        return rect

    # Calculate row-by-row density (fraction of dark/content pixels per row)
    row_density = np.sum(region < 128, axis=1) / w

    # Smooth lightly to reduce single-pixel noise but preserve real gaps
    kernel_size = max(3, h // 50)
    if kernel_size % 2 == 0:
        kernel_size += 1
    smoothed = np.convolve(row_density, np.ones(kernel_size) / kernel_size, mode='same')

    # Strategy: scan bottom-up looking for a whitespace band (consecutive rows
    # with very low density spanning the full width). This band separates the
    # panel artwork above from caption text below.
    #
    # On every storyboard, there's a visible gap between the drawing area and
    # the text caption — typically 3-15px of near-empty rows.

    gap_threshold = 0.02  # rows with <= 2% dark pixels are "whitespace"
    min_gap_rows = 3      # need at least 3 consecutive whitespace rows

    # Don't look in the top 40% — the gap is always in the lower portion
    search_top = int(h * 0.4)

    # Scan bottom-up to find the LOWEST whitespace band
    # (this is the drawing/text boundary, not internal whitespace in the drawing)
    best_gap_top = None
    gap_count = 0

    for row_idx in range(h - 1, search_top - 1, -1):
        if smoothed[row_idx] <= gap_threshold:
            gap_count += 1
        else:
            if gap_count >= min_gap_rows:
                # Found a whitespace band; its top edge is where we trim
                best_gap_top = row_idx + 1
                break
            gap_count = 0

    # Edge case: gap extends to or near search_top
    if gap_count >= min_gap_rows and best_gap_top is None:
        best_gap_top = search_top

    if best_gap_top is not None and best_gap_top > h * 0.4:
        # Verify there's actual content (text) below the gap — if not, this
        # might be trailing whitespace and trimming is still fine
        below_density = np.mean(smoothed[best_gap_top:]) if best_gap_top < h else 0

        # Only trim if: there IS text below, OR we're trimming trailing whitespace
        # (both are desirable — we want just the drawing)
        print(f"[Trim] rect at y={y}: gap found at row {best_gap_top}/{h}, "
              f"below_density={below_density:.3f}", file=sys.stderr, flush=True)

        return {
            'x': rect['x'], 'y': rect['y'],
            'width': rect['width'], 'height': best_gap_top,
            'area': rect.get('area', rect['width'] * best_gap_top),
            'fill': rect.get('fill', 0.5)
        }

    return rect


def detect_from_projection_profile(img_gray, inverted, width, height, img_area, expected_count):
    """
    Detect panels using horizontal and vertical projection profiles.
    Works for photo boards where morphology fails because photos become
    fragmented "Swiss cheese" in the binary mask.

    Approach:
    1. Compute horizontal projection (row sums) to find row boundaries
       — rows of panels are separated by whitespace bands (which contain text)
    2. Within each row, compute vertical projection to find column boundaries
    3. Build panel rectangles from the grid intersections

    This works because even fragmented photo content has significantly more
    dark pixels per row than the gap/text rows between panel rows.
    """
    # Horizontal projection: sum of white pixels per row (inverted = white=content)
    h_proj = np.sum(inverted > 128, axis=1).astype(float) / width

    # Smooth to find broad row bands
    smooth_k = max(5, height // 60)
    if smooth_k % 2 == 0:
        smooth_k += 1
    h_smooth = np.convolve(h_proj, np.ones(smooth_k) / smooth_k, mode='same')

    # Find horizontal gap bands: rows where content density drops significantly
    # These gaps contain text captions between panel rows
    gap_threshold = 0.10  # rows with <10% content are gaps
    min_gap_height = max(5, height // 80)
    min_content_height = height // 12

    # Identify content rows vs gap rows
    is_content = h_smooth > gap_threshold

    # Find content bands (runs of content rows)
    content_bands = []
    in_band = False
    band_start = 0
    for i in range(height):
        if is_content[i] and not in_band:
            band_start = i
            in_band = True
        elif not is_content[i] and in_band:
            if i - band_start >= min_content_height:
                content_bands.append((band_start, i))
            in_band = False
    if in_band and height - band_start >= min_content_height:
        content_bands.append((band_start, height))

    if len(content_bands) < 1:
        return []

    print(f"[Projection] Found {len(content_bands)} content rows", file=sys.stderr, flush=True)

    # For each content band, find column boundaries using vertical projection
    all_panels = []
    for band_top, band_bottom in content_bands:
        band_slice = inverted[band_top:band_bottom, :]
        v_proj = np.sum(band_slice > 128, axis=0).astype(float) / (band_bottom - band_top)

        # Smooth vertically
        v_smooth_k = max(5, width // 60)
        if v_smooth_k % 2 == 0:
            v_smooth_k += 1
        v_smooth = np.convolve(v_proj, np.ones(v_smooth_k) / v_smooth_k, mode='same')

        # Find vertical gaps (columns between panels)
        v_gap_threshold = 0.08
        min_v_gap = max(5, width // 100)
        min_panel_width = width // 12

        is_panel_col = v_smooth > v_gap_threshold

        # Find panel column spans
        panels_in_row = []
        in_panel = False
        panel_start = 0
        for col in range(width):
            if is_panel_col[col] and not in_panel:
                panel_start = col
                in_panel = True
            elif not is_panel_col[col] and in_panel:
                if col - panel_start >= min_panel_width:
                    panels_in_row.append((panel_start, col))
                in_panel = False
        if in_panel and width - panel_start >= min_panel_width:
            panels_in_row.append((panel_start, width))

        for col_left, col_right in panels_in_row:
            all_panels.append({
                'x': int(col_left), 'y': int(band_top),
                'width': int(col_right - col_left),
                'height': int(band_bottom - band_top),
                'area': float((col_right - col_left) * (band_bottom - band_top)),
                'fill': 0.5
            })

    print(f"[Projection] Found {len(all_panels)} panels total", file=sys.stderr, flush=True)
    return all_panels


def detect_from_mask(mask_path, expected_count=0):
    """
    Detect panels from a pre-thresholded binary mask image.
    The mask has black content on white background (as generated by sharp.threshold()).

    Detection strategies tried in order:
    1. Projection profile — finds panel grid from row/column density profiles.
       Works for photo boards where content is fragmented but rows are distinct.
    2. Morphology strategies — vertical erosion to kill text, close to fill holes.
       Works when panels are solid blobs (hand-drawn boards with clean masks).
    3. Light strategy — no erosion, for already-clean masks.
    """
    img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []

    height, width = img.shape[:2]
    img_area = width * height

    # Invert: white content on black background
    inverted = cv2.bitwise_not(img)

    best_candidates = []
    best_strategy = "none"

    # === Strategy 0: Projection profile (grid geometry from density) ===
    # Try this first — it handles photo boards that morphology can't
    proj_candidates = detect_from_projection_profile(img, inverted, width, height, img_area, expected_count)
    if proj_candidates:
        min_needed = max(1, int(expected_count * 0.7)) if expected_count > 0 else 1
        if len(proj_candidates) >= min_needed:
            best_candidates = proj_candidates
            best_strategy = "projection"
            print(f"[Mask] Strategy 'projection': found {len(proj_candidates)} candidates (expected ~{expected_count})", file=sys.stderr, flush=True)
            if expected_count > 0 and len(proj_candidates) == expected_count:
                # Perfect match — use it
                best_candidates = [trim_caption_text(img, c) for c in best_candidates]
                best_candidates = finalize_candidates(best_candidates, expected_count, height)
                return best_candidates

    # === Strategies 1-4: Morphology-based ===
    for strategy_name, erode_h, close_size, close_iter in [
        ("erode-20", 20, 30, 3),    # kill text <=20px tall, aggressive close
        ("erode-15", 15, 25, 3),    # slightly less aggressive
        ("erode-12", 12, 20, 2),    # lighter still
        ("light",     0,  5, 1),    # no text removal (for already-clean masks)
    ]:
        working = inverted.copy()

        # Step 1: Vertical erosion to kill text lines
        if erode_h > 0:
            v_erode = np.ones((erode_h, 1), np.uint8)
            working = cv2.erode(working, v_erode, iterations=1)
            # Dilate back to restore vertical extent
            working = cv2.dilate(working, v_erode, iterations=1)

        # Step 2: Close to fill holes inside photos
        if close_size > 0:
            close_kernel = np.ones((close_size, close_size), np.uint8)
            working = cv2.morphologyEx(working, cv2.MORPH_CLOSE, close_kernel, iterations=close_iter)

        # Step 3: Open to re-separate panels that may have merged
        # Horizontal open separates columns, vertical open separates rows
        h_open = np.ones((1, 20), np.uint8)
        working = cv2.morphologyEx(working, cv2.MORPH_OPEN, h_open, iterations=1)
        v_open = np.ones((15, 1), np.uint8)
        working = cv2.morphologyEx(working, cv2.MORPH_OPEN, v_open, iterations=1)

        candidates = find_panel_contours(working, width, height, img_area, expected_count)

        print(f"[Mask] Strategy '{strategy_name}': found {len(candidates)} candidates (expected ~{expected_count})", file=sys.stderr, flush=True)

        # Check if this is good enough
        min_needed = max(1, int(expected_count * 0.7)) if expected_count > 0 else 1
        if len(candidates) >= min_needed:
            if len(candidates) > len(best_candidates):
                best_candidates = candidates
                best_strategy = strategy_name
            # If we hit exact count, stop trying
            if expected_count > 0 and len(candidates) == expected_count:
                break

    if not best_candidates:
        return []

    print(f"[Mask] Using strategy '{best_strategy}' with {len(best_candidates)} panels", file=sys.stderr, flush=True)

    # Trim caption text from each panel using the ORIGINAL mask
    best_candidates = [trim_caption_text(img, c) for c in best_candidates]
    best_candidates = finalize_candidates(best_candidates, expected_count, height)

    return best_candidates


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