"""Measure the fixed camera's A4 paper plane without commanding the arm."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def marker_candidates(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, _, stats, centers = cv2.connectedComponentsWithStats(cv2.inRange(gray, 0, 70))
    return [
        {"center_px": [round(float(n), 2) for n in center], "area_px": int(area)}
        for (x, y, width, height, area), center in zip(stats[1:], centers[1:], strict=True)
        if 4 <= area <= 90 and 2 <= width <= 15 and 2 <= height <= 15
        and 65 <= x <= 410 and 150 <= y <= 469
    ]


def map_points(matrix, points):
    pixels = np.asarray(points, dtype=np.float32).reshape(1, -1, 2)
    return cv2.perspectiveTransform(pixels, matrix)[0]


def measure_paper_corners(frame, guides):
    gray = cv2.cvtColor(frame.astype(np.float32), cv2.COLOR_BGR2GRAY)
    signal = cv2.GaussianBlur(-gray, (3, 3), 0.75)
    center = guides.mean(axis=0)
    lines = []
    for start, end in zip(guides, np.roll(guides, -1, axis=0), strict=True):
        edge = end - start
        inward = np.array([-edge[1], edge[0]], dtype=np.float32) / np.linalg.norm(edge)
        if inward @ (center - (start + end) * 0.5) < 0:
            inward = -inward
        along = np.linspace(0.18, 0.82, 96, dtype=np.float32)
        across = np.arange(-10, 11, dtype=np.float32)
        stripe = start + along[:, None, None] * edge + across[None, :, None] * inward
        values = cv2.remap(signal, stripe[:, :, 0].copy(), stripe[:, :, 1].copy(), cv2.INTER_LINEAR)
        changes = values[:, 2:] - values[:, :-2]
        best = changes.argmax(axis=1)
        strengths = changes[np.arange(len(along)), best]
        points = stripe[np.arange(len(along)), best + 1][strengths >= max(2, np.percentile(strengths, 30))]
        if len(points) < 20:
            raise ValueError("Could not find a clear paper edge near the corner markers")
        line = cv2.fitLine(points, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        residuals = np.abs((points - line[2:]) @ [-line[1], line[0]])
        if np.median(residuals) > 2:
            raise ValueError("The paper edge has an inconsistent line fit")
        lines.append(line)

    corners = []
    for preceding, following in zip(lines[-1:] + lines[:-1], lines, strict=True):
        distances = np.linalg.solve(
            np.column_stack((preceding[:2], -following[:2])), following[2:] - preceding[2:],
        )
        corners.append(preceding[2:] + distances[0] * preceding[:2])
    corners = np.asarray(corners, dtype=np.float32)
    if np.max(np.linalg.norm(corners - guides, axis=1)) > 15:
        raise ValueError("A fitted paper corner is too far from its tape marker")
    return corners


def build_plane(frame, corners, candidate_point):
    height, width = frame.shape[:2]
    if not ((corners >= 0) & (corners < [width, height])).all():
        raise ValueError("All four paper corners must be inside the reference frame")
    if not cv2.isContourConvex(corners) or abs(cv2.contourArea(corners)) < 1_000:
        raise ValueError("Expected four ordered paper corners spanning a convex area")
    paper = np.array([(0, 0), (297, 0), (297, 210), (0, 210)], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(corners, paper, cv2.DECOMP_SVD)
    projected = map_points(matrix, corners)
    if not np.isfinite(matrix).all() or np.max(np.abs(projected - paper)) > 0.1:
        raise ValueError("Perspective fit is inconsistent")
    result = {
        "camera_id": "SO101_CAMERA",
        "image_size_px": [width, height],
        "reference": "A4 paper",
        "reference_plane_mm": [297, 210],
        "corners_px": corners.tolist(),
        "corners_mm": paper.tolist(),
        "pixel_to_paper_mm": matrix.tolist(),
        "robot_base_registration": None,
        "independent_scale_check": None,
    }
    if candidate_point is not None:
        projected = map_points(matrix, [candidate_point])[0]
        result["example"] = {
            "image_point_px": candidate_point,
            "ray_intersection_with_paper_mm": [round(float(n), 1) for n in projected],
            "inside_paper": 0 <= projected[0] <= 297 and 0 <= projected[1] <= 210,
        }
        result["example"]["inside_paper"] = bool(result["example"]["inside_paper"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--corners", type=float, nargs=8, metavar="PIXEL",
                        help="Four vertices in order, starting with a 297 mm edge")
    parser.add_argument("--point", type=float, nargs=2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inferred-corner", type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--repeat-image", type=Path, help="Refine the edges and compare two fixed-camera frames")
    args = parser.parse_args()
    frame = cv2.imread(str(args.image))
    if frame is None:
        parser.error(f"Could not read {args.image}")
    if args.corners is None:
        print(json.dumps(marker_candidates(frame), indent=2))
        return
    corners = np.asarray(args.corners, dtype=np.float32).reshape(4, 2)
    repeatability = None
    if args.repeat_image:
        repeat = cv2.imread(str(args.repeat_image))
        if repeat is None or repeat.shape != frame.shape:
            parser.error("The repeat frame must be readable and match the original resolution")
        first_corners = measure_paper_corners(frame, corners)
        repeat_corners = measure_paper_corners(repeat, corners)
        repeatability = float(np.max(np.linalg.norm(first_corners - repeat_corners, axis=1)))
        if repeatability > 2.5:
            raise ValueError(f"Paper corner repeatability exceeds 2.5 px: {repeatability:.2f} px")
        corners = (first_corners + repeat_corners) * 0.5
    result = build_plane(frame, corners, args.point)
    result["inferred_corner"] = args.inferred_corner
    result["max_corner_difference_between_frames_px"] = None if repeatability is None else round(repeatability, 2)
    data = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(data + "\n", encoding="utf-8")
    print(data)


if __name__ == "__main__":
    main()
