from __future__ import annotations

from collections.abc import Sequence


Point = tuple[float, float]
BBox = tuple[float, float, float, float]


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Ray-casting point-in-polygon test, including boundary as inside."""
    x, y = point
    inside = False
    n = len(polygon)
    if n < 3:
        return False

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if _point_on_segment(point, (xi, yi), (xj, yj)):
            return True
        intersects = (yi > y) != (yj > y)
        if intersects:
            x_at_y = (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
            if x <= x_at_y:
                inside = not inside
        j = i
    return inside


def bbox_anchor_point(bbox: BBox, footpoint_ratio: float = 0.92) -> Point:
    x1, y1, x2, y2 = bbox
    x = (x1 + x2) / 2.0
    y = y1 + (y2 - y1) * footpoint_ratio
    return x, y


def _point_on_segment(point: Point, start: Point, end: Point, eps: float = 1e-6) -> bool:
    px, py = point
    x1, y1 = start
    x2, y2 = end
    cross = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
    if abs(cross) > eps:
        return False
    return min(x1, x2) - eps <= px <= max(x1, x2) + eps and min(y1, y2) - eps <= py <= max(y1, y2) + eps
