from __future__ import annotations

from manga_scanner.types import BoundingBox

# Manga reading order: right-to-left within a row, rows top-to-bottom.
# We define a "row" as boxes whose vertical centers fall within ROW_TOLERANCE
# pixels of each other. Within each row, sort by x_center descending (right first).
_ROW_TOLERANCE = 40


def sort_reading_order(boxes: list[BoundingBox]) -> list[BoundingBox]:
    """
    Returns boxes sorted in manga reading order.
    Rows are identified by grouping boxes with similar y_center values,
    then sorted top-to-bottom. Within each row, boxes are sorted
    right-to-left (descending x_center).
    """
    if not boxes:
        return []

    sorted_by_y = sorted(boxes, key=lambda b: b.y_center)

    rows: list[list[BoundingBox]] = []
    current_row: list[BoundingBox] = [sorted_by_y[0]]

    for box in sorted_by_y[1:]:
        if abs(box.y_center - current_row[-1].y_center) <= _ROW_TOLERANCE:
            current_row.append(box)
        else:
            rows.append(current_row)
            current_row = [box]
    rows.append(current_row)

    result: list[BoundingBox] = []
    for row in rows:
        result.extend(sorted(row, key=lambda b: b.x_center, reverse=True))

    return result
