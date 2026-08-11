from edge_inspection.geometry import bbox_anchor_point, point_in_polygon


def test_point_in_polygon_inside_and_boundary() -> None:
    polygon = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert point_in_polygon((5, 5), polygon)
    assert point_in_polygon((0, 5), polygon)
    assert not point_in_polygon((12, 5), polygon)


def test_bbox_anchor_point_uses_footpoint_ratio() -> None:
    assert bbox_anchor_point((0, 0, 10, 100), 0.9) == (5, 90)
