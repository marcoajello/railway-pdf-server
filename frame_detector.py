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
import os
import base64
import subprocess


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

def erase_text_from_mask(img, mask_path):
    """
    Run tesseract OCR on the mask to find all text regions, then paint them
    white (background) on the mask.
    Uses multiple PSM modes for better coverage.

    Returns: (cleaned_mask, text_boxes)
      - cleaned_mask: mask with text regions painted white
      - text_boxes: list of {x, y, w, h} dicts for every detected text region
    """
    cleaned = img.copy()
    text_boxes = {}
    pad = 3

    for psm in ['3', '11', '6']:
        try:
            result = subprocess.run(
                ['tesseract', mask_path, 'stdout', '--psm', psm, 'tsv'],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                continue

            lines = result.stdout.strip().split('\n')
            for line in lines[1:]:
                fields = line.split('\t')
                if len(fields) < 12:
                    continue
                try:
                    conf = float(fields[10])
                    text = fields[11].strip()
                except (ValueError, IndexError):
                    continue
                if conf < 25 or not text:
                    continue

                left = max(0, int(fields[6]) - pad)
                top = max(0, int(fields[7]) - pad)
                w = int(fields[8]) + pad * 2
                h_box = int(fields[9]) + pad * 2

                # Deduplicate by position
                key = (left // 5, top // 5)
                if key not in text_boxes:
                    cleaned[top:top + h_box, left:left + w] = 255
                    text_boxes[key] = {'x': left, 'y': top, 'w': w, 'h': h_box}

        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    box_list = list(text_boxes.values())
    print(f"[OCR] Erased {len(box_list)} text regions from mask",
          file=sys.stderr, flush=True)
    return cleaned, box_list


def sample_background_color(img, h, w):
    """
    Sample background color from image edges (strips, not corners).
    Corners may be artwork; edge strips are more reliable.
    """
    strip = max(3, min(h, w) // 30)
    samples = []
    # Top and bottom edge strips
    samples.append(img[0:strip, :].reshape(-1, 3))
    samples.append(img[h - strip:, :].reshape(-1, 3))
    # Left and right edge strips
    samples.append(img[:, 0:strip].reshape(-1, 3))
    samples.append(img[:, w - strip:].reshape(-1, 3))
    all_samples = np.vstack(samples)
    return np.median(all_samples, axis=0).astype(int).tolist()


def trim_caption_from_crop(img):
    """
    Content-density bottom trim: find where artwork ends and caption
    text/whitespace begins by scanning horizontal projection from the
    bottom up.

    Returns trimmed image (or original if no caption zone found).
    """
    h, w = img.shape[:2]
    if h < 60 or w < 60:
        return img

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Compute per-row content density (fraction of non-background pixels)
    # Use adaptive threshold to handle varying background brightness
    bg_value = int(np.median(gray[0:3, :]))  # top edge → likely background
    # Pixels significantly darker than background = content
    threshold = max(50, bg_value - 40)
    content_mask = gray < threshold

    # Horizontal projection: fraction of content pixels per row
    row_density = np.sum(content_mask, axis=1).astype(float) / w

    # Smooth with small kernel
    smooth_k = max(3, h // 40)
    if smooth_k % 2 == 0:
        smooth_k += 1
    smoothed = np.convolve(row_density, np.ones(smooth_k) / smooth_k, mode='same')

    # Find the artwork/caption boundary by scanning from bottom up.
    # Caption zone: rows with low density (< 15% content — sparse text on background)
    # Artwork zone: rows with higher density (> 25% content — illustrations)
    #
    # We look for the transition: a sustained low-density zone at the bottom
    # that's at least 5% of image height.
    caption_threshold = 0.15
    artwork_threshold = 0.25
    min_caption_height = max(10, int(h * 0.05))

    # Scan from bottom up to find where caption zone starts
    caption_start = h  # default: no caption found
    low_run = 0
    for row in range(h - 1, int(h * 0.40), -1):  # don't trim more than 60%
        if smoothed[row] < caption_threshold:
            low_run += 1
        elif smoothed[row] > artwork_threshold:
            # Hit artwork — if we accumulated enough low-density rows, mark boundary
            if low_run >= min_caption_height:
                caption_start = row + 1
            break
        else:
            # Ambiguous zone — keep scanning
            pass

    # Also check: if the bottom 15% is nearly all background, trim it
    if caption_start == h:
        bottom_zone = smoothed[int(h * 0.85):]
        if len(bottom_zone) > 0 and np.mean(bottom_zone) < 0.08:
            caption_start = int(h * 0.85)

    if caption_start < h and caption_start > int(h * 0.40):
        trim_amount = h - caption_start
        print(f"[Trim] Trimming {trim_amount}px caption zone from bottom "
              f"({trim_amount * 100 // h}% of height)",
              file=sys.stderr, flush=True)
        return img[0:caption_start, :]

    return img


def run_tesseract_multi(image_path, timeout=15):
    """
    Run Tesseract with multiple PSM modes and union the results.
    PSM 6 (uniform block) + PSM 11 (sparse text) catches more caption styles.
    Returns list of (left, top, width, height, confidence, text) tuples.
    """
    all_boxes = {}

    for psm in ['6', '11']:
        try:
            result = subprocess.run(
                ['tesseract', image_path, 'stdout', '--psm', psm, 'tsv'],
                capture_output=True, text=True, timeout=timeout
            )
            if result.returncode != 0:
                continue

            lines = result.stdout.strip().split('\n')
            for line in lines[1:]:
                fields = line.split('\t')
                if len(fields) < 12:
                    continue
                try:
                    conf = float(fields[10])
                    text = fields[11].strip()
                except (ValueError, IndexError):
                    continue
                if conf < 25 or not text:
                    continue

                left = int(fields[6])
                top = int(fields[7])
                bw = int(fields[8])
                bh = int(fields[9])

                # Deduplicate by position (within 5px tolerance)
                key = (left // 5, top // 5)
                existing = all_boxes.get(key)
                if not existing or conf > existing[4]:
                    all_boxes[key] = (left, top, bw, bh, conf, text)

        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    return list(all_boxes.values())


def erase_text_from_crop(image_path):
    """
    Erase caption text from a single cropped panel image.
    Two-phase approach:
      1. Content-density trim: detect where artwork ends and caption
         whitespace/text begins, crop it off entirely
      2. OCR cleanup: run multi-pass Tesseract on the remaining image
         to catch any residual text in the bottom 35%

    Returns base64-encoded JPEG of the cleaned image, or None if no changes.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    orig_h, orig_w = img.shape[:2]

    # Phase 1: Content-density trim (removes obvious caption zones)
    trimmed = trim_caption_from_crop(img)
    was_trimmed = trimmed.shape[0] < orig_h

    # Phase 2: OCR-based erasure on the (possibly trimmed) image
    h, w = trimmed.shape[:2]
    lower_threshold = int(h * 0.65)  # only erase text in bottom 35%

    # Write trimmed image for Tesseract
    if was_trimmed:
        tmp_path = image_path + '.trimmed.jpg'
        cv2.imwrite(tmp_path, trimmed)
        ocr_path = tmp_path
    else:
        ocr_path = image_path

    try:
        boxes = run_tesseract_multi(ocr_path, timeout=15)

        bg_color = sample_background_color(trimmed, h, w)

        pad = 4
        erased = 0
        for (left, top, bw, bh, conf, text) in boxes:
            if top < lower_threshold:
                continue  # skip text in upper portion (part of artwork)

            x1 = max(0, left - pad)
            y1 = max(0, top - pad)
            x2 = min(w, left + bw + pad)
            y2 = min(h, top + bh + pad)

            trimmed[y1:y2, x1:x2] = bg_color
            erased += 1

        # Clean up temp file
        if was_trimmed:
            try:
                os.remove(ocr_path)
            except OSError:
                pass

        if not was_trimmed and erased == 0:
            return None  # no changes made

        changes = []
        if was_trimmed:
            changes.append(f"trimmed {orig_h - h}px")
        if erased > 0:
            changes.append(f"erased {erased} text regions")
        print(f"[OCR-crop] {', '.join(changes)}", file=sys.stderr, flush=True)

        _, buf = cv2.imencode('.jpg', trimmed, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf).decode('ascii')

    except (FileNotFoundError, subprocess.TimeoutExpired):
        if was_trimmed:
            # Still return the trimmed version even if OCR failed
            _, buf = cv2.imencode('.jpg', trimmed, [cv2.IMWRITE_JPEG_QUALITY, 85])
            print(f"[OCR-crop] trimmed {orig_h - h}px (OCR unavailable)",
                  file=sys.stderr, flush=True)
            return base64.b64encode(buf).decode('ascii')
        return None


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

    # Refine each content band: trim bottom text/caption zone
    # Caption text below panels has lower, more uniform density than artwork
    refined_bands = []
    for band_top, band_bottom in content_bands:
        band_h = band_bottom - band_top
        if band_h < min_content_height:
            refined_bands.append((band_top, band_bottom))
            continue

        # Compute fine-grained horizontal projection within this band
        band_slice = inverted[band_top:band_bottom, :]
        band_proj = np.sum(band_slice > 128, axis=1).astype(float) / width

        # Smooth lightly
        bk = max(3, band_h // 30)
        if bk % 2 == 0:
            bk += 1
        band_smooth = np.convolve(band_proj, np.ones(bk) / bk, mode='same')

        # Scan from bottom of band upward to find where text zone begins
        # Text zone: sustained low density (< 12%) at bottom of band
        text_thresh = 0.12
        art_thresh = 0.20
        min_text_zone = max(8, band_h // 10)

        trim_row = band_h  # relative to band_top
        low_run = 0
        for row in range(band_h - 1, band_h // 3, -1):
            if band_smooth[row] < text_thresh:
                low_run += 1
            elif band_smooth[row] > art_thresh:
                if low_run >= min_text_zone:
                    trim_row = row + 1
                break
            # else ambiguous, keep scanning

        if trim_row < band_h:
            print(f"[Projection] Band {band_top}-{band_bottom}: trimmed "
                  f"{band_h - trim_row}px caption zone from bottom",
                  file=sys.stderr, flush=True)
            refined_bands.append((band_top, band_top + trim_row))
        else:
            refined_bands.append((band_top, band_bottom))

    # For each refined content band, find column boundaries using vertical projection
    all_panels = []
    for band_top, band_bottom in refined_bands:
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

    Step 0: OCR text erasure — run tesseract on the mask to find text regions,
            paint them white. All subsequent detection runs on a TEXT-FREE mask.

    Then detection strategies tried in order:
    1. Projection profile — finds panel grid from row/column density profiles.
    2. Morphology strategies — vertical erosion to kill remaining noise, close holes.
    3. Light strategy — no erosion, for already-clean masks.
    """
    img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return [], []

    height, width = img.shape[:2]
    img_area = width * height

    # === Step 0: Erase text from mask using OCR ===
    # The mask has black text on white background — tesseract finds it well.
    # text_boxes are preserved for reference.
    cleaned, text_boxes = erase_text_from_mask(img, mask_path)

    # Invert cleaned mask: white content on black background
    inverted = cv2.bitwise_not(cleaned)

    best_candidates = []
    best_strategy = "none"

    # === Strategy 1: Projection profile (grid geometry from density) ===
    # Try this first — it handles photo boards that morphology can't
    proj_candidates = detect_from_projection_profile(cleaned, inverted, width, height, img_area, expected_count)
    if proj_candidates:
        min_needed = max(1, int(expected_count * 0.7)) if expected_count > 0 else 1
        if len(proj_candidates) >= min_needed:
            best_candidates = proj_candidates
            best_strategy = "projection"
            print(f"[Mask] Strategy 'projection': found {len(proj_candidates)} candidates (expected ~{expected_count})", file=sys.stderr, flush=True)
            if expected_count > 0 and len(proj_candidates) == expected_count:
                # Perfect match — use it (no trim needed, text already erased)
                best_candidates = finalize_candidates(best_candidates, expected_count, height)
                return best_candidates, text_boxes

    # === Strategies 2-5: Morphology-based ===
    for strategy_name, erode_h, close_size, close_iter in [
        ("erode-20", 20, 30, 3),    # kill remaining noise <=20px tall, aggressive close
        ("erode-15", 15, 25, 3),    # slightly less aggressive
        ("erode-12", 12, 20, 2),    # lighter still
        ("light",     0,  5, 1),    # no erosion (for already-clean masks)
    ]:
        working = inverted.copy()

        # Step 1: Vertical erosion to kill thin noise
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
        h_open = np.ones((1, 20), np.uint8)
        working = cv2.morphologyEx(working, cv2.MORPH_OPEN, h_open, iterations=1)
        v_open = np.ones((15, 1), np.uint8)
        working = cv2.morphologyEx(working, cv2.MORPH_OPEN, v_open, iterations=1)

        candidates = find_panel_contours(working, width, height, img_area, expected_count)

        print(f"[Mask] Strategy '{strategy_name}': found {len(candidates)} candidates (expected ~{expected_count})", file=sys.stderr, flush=True)

        min_needed = max(1, int(expected_count * 0.7)) if expected_count > 0 else 1
        if len(candidates) >= min_needed:
            if len(candidates) > len(best_candidates):
                best_candidates = candidates
                best_strategy = strategy_name
            if expected_count > 0 and len(candidates) == expected_count:
                break

    if not best_candidates:
        return [], text_boxes

    print(f"[Mask] Using strategy '{best_strategy}' with {len(best_candidates)} panels", file=sys.stderr, flush=True)

    # No trim_caption_text needed — text was erased from mask before detection
    best_candidates = finalize_candidates(best_candidates, expected_count, height)

    return best_candidates, text_boxes


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
        if mode_arg == 'erase_text':
            # Erase text mode: run OCR on a single cropped panel image,
            # find text in the bottom 40%, paint it with background color.
            # Returns base64 JPEG of cleaned image, or null if no text found.
            cleaned_b64 = erase_text_from_crop(image_path)
            result = {'cleaned': cleaned_b64}
            print(json.dumps(result))

        elif mode_arg == 'mask':
            # Mask mode: detect panels from pre-thresholded mask image
            # argv[3] = expected count (optional)
            # Returns rectangles only — Node handles full-res cropping
            expected = int(sys.argv[3]) if len(sys.argv) > 3 else 0

            rects, ocr_text_boxes = detect_from_mask(image_path, expected)
            result = {
                'count': len(rects),
                'mode': 'mask',
                'rectangles': rects,
                'textBoxes': ocr_text_boxes
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