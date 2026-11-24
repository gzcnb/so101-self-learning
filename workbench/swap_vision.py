"""Vision on the calibrated SO-101 work surface."""

import cv2
import numpy as np

from supervise_snapshot import find_orange_target


def find_dark_parts(image_path, view):
    frame = cv2.imread(str(image_path))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, 0, 91) & cv2.inRange(hsv[:, :, 1], 0, 121)
    _, _, stats, centers = cv2.connectedComponentsWithStats(mask)
    candidates = []
    low, high = (100, 2400) if view == 'fixed' else (1000, 43000)
    for box, center in zip(stats[1:], centers[1:], strict=True):
        x, y, width, height, area = (int(value) for value in box)
        if low <= area <= high and 0.55 <= width / height <= 1.55 and area / (width * height) >= 0.48:
            candidates.append({'box_px': [x, y, width, height], 'center_px': center.round(2).tolist()})
    return candidates


def match_support_edge(reference_path, image_path, support):
    """Confirm the visible lower edge of a support whose top is occluded."""
    reference = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    x, y, width, height = support['box_px']
    left, right = x - 5, x + width + 5
    top, bottom = y + round(height * 0.65), y + height + 6
    margin = 5
    if (reference is None or image is None or left < margin or top < margin
            or right + margin >= image.shape[1] or bottom + margin >= image.shape[0]):
        return None
    known = reference[top:bottom, left:right]
    window = image[top - margin:bottom + margin, left - margin:right + margin]
    scores = cv2.matchTemplate(window, known, cv2.TM_CCOEFF_NORMED)
    _, score, _, (i, j) = cv2.minMaxLoc(scores)
    dx, dy = i - margin, j - margin
    candidate = image[top + dy:bottom + dy, left + dx:right + dx]
    known_dark, seen_dark = known < 103, candidate < 103
    intersection = np.count_nonzero(known_dark & seen_dark)
    union = np.count_nonzero(known_dark | seen_dark)
    if score < 0.78 or intersection < 50 or intersection / union < 0.71:
        return None
    cx, cy = support['center_px']
    return {'center_px': [cx + dx, cy + dy], 'box_px': [x + dx, y + dy, width, height],
            'edge_match': round(score, 3)}


def pixel_to_table(point, homography):
    return cv2.perspectiveTransform(np.array([[point]], dtype=np.float64), homography)[0, 0]


def identify(pairs, homography):
    """Identify a unique stable candidate per color from pairs of fixed-camera frames."""
    result = {}
    for color, detector in [('orange', find_orange_target), ('dark_gray', find_dark_parts)]:
        stable = []
        for before, after in pairs:
            options = []
            for path in (before, after):
                eligible = []
                for item in detector(path, 'fixed'):
                    u, v = pixel_to_table(item['center_px'], homography)
                    _, _, width, height = item['box_px']
                    if 22 <= u <= 270 and 30 <= v <= 179 and 15 <= width <= 46 and 17 <= height <= 48:
                        eligible.append(item['center_px'])
                options.append(eligible)
            for center in options[1]:
                if any(np.linalg.norm(np.subtract(center, other)) < 2.8 for other in options[0]):
                    stable.append(np.array(center))
        if not stable or any(np.linalg.norm(center - stable[-1]) > 10 for center in stable):
            raise RuntimeError(f'无法在纸面上唯一确认{ "橙色" if color == "orange" else "深灰色" }方块')
        result[color] = stable[-1].tolist()
    return result


def temporary_candidates(pairs, scene, homography):
    frames = [cv2.imread(str(pair[1])) for pair in pairs]
    objects = [pixel_to_table(center, homography) for center in scene.values()]
    ranked = []
    for y in range(266, 358, 9):
        for x in range(328, 370, 7):
            point = (x, y)
            u, v = pixel_to_table(point, homography)
            if not (42 < u < 254 and 102 < v < 179) or min(np.linalg.norm([u, v] - block) for block in objects) < 49:
                continue
            for frame in frames:
                patch = frame[y - 12:y + 12, x - 12:x + 12]
                hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                levels = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                if levels.min() >= 125 and np.std(levels) <= 9 and not (hsv[:, :, 1] > 95).any():
                    ranked.append((np.linalg.norm(np.subtract(point, (346, 310))), point))
                    break
    return [point for _, point in sorted(ranked)]
