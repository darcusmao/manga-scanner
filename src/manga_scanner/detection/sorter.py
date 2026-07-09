from __future__ import annotations

from manga_scanner.types import BoundingBox


def sort_reading_order(
    boxes: list[BoundingBox],
    row_threshold: int = 50,
) -> list[BoundingBox]:
    """
    Returns boxes sorted in manga reading order: top-to-bottom rows,
    right-to-left within each row.

    row_threshold: maximum y_center distance (pixels) for two boxes to be
                   considered part of the same row.
    """
    if not boxes:
        return []

    sorted_by_y = sorted(boxes, key=lambda b: b.y_center)

    rows: list[list[BoundingBox]] = []
    current_row: list[BoundingBox] = [sorted_by_y[0]]
    current_row_y = sorted_by_y[0].y_center

    for box in sorted_by_y[1:]:
        if abs(box.y_center - current_row_y) <= row_threshold:
            current_row.append(box)
        else:
            rows.append(current_row)
            current_row = [box]
            current_row_y = box.y_center
    rows.append(current_row)

    result: list[BoundingBox] = []
    for row in rows:
        result.extend(sorted(row, key=lambda b: b.x_center, reverse=True))

    return result
