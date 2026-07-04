# TICKET-006: Manga Reading Order Sorting

## Summary
Write a function that takes a list of `BoundingBox` objects and reorders them into manga reading order: right-to-left within each row, rows sorted top-to-bottom. This ordering is passed to the LLM so translated strings map to the correct speech bubble positions and the LLM receives dialogue in coherent conversational sequence.

## Language and Tools
- Python 3.11 standard library only (no additional packages)

## Why This Is Non-Trivial
Bounding boxes from YOLOv8 are returned in detection confidence order, not spatial order. If we pass `["text from bottom-left", "text from top-right"]` to the LLM for a two-bubble page, the translations are in wrong conversational order, and when mapped back to positions the dialogue reads nonsensically.

## Algorithm

Manga reading order for a page:
1. Rows flow top-to-bottom (ascending Y)
2. Within each row, bubbles read right-to-left (descending X)

Two bubbles belong to the same "row" if their Y-centroids are within `row_threshold` pixels of each other. The threshold must be larger than typical intra-row vertical misalignment but smaller than the gap between panel rows.

```
Row grouping logic:
  Sort all boxes by y_center ascending.
  Walk sorted list; if next box's y_center is within row_threshold of the current row's 
  representative y_center, add it to the current row. Otherwise, start a new row.
  Within each row, sort by x_center descending (right-to-left).
```

File: `src/manga_scanner/detection/sorter.py`

```python
from manga_scanner.types import BoundingBox


def sort_reading_order(
    boxes: list[BoundingBox],
    row_threshold: int = 50,
) -> list[BoundingBox]:
    """
    Returns boxes sorted in Japanese manga reading order:
    top-to-bottom rows, right-to-left within each row.

    row_threshold: maximum y_center distance (pixels) for two boxes
                   to be considered part of the same row.
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

    result = []
    for row in rows:
        result.extend(sorted(row, key=lambda b: b.x_center, reverse=True))

    return result
```

## Threshold Tuning

The default `row_threshold=50` pixels works for typical manga at 1200-2000px page height. At higher resolutions, increase proportionally. This value should be exposed in `DetectionConfig` if users frequently process non-standard scan resolutions:

```yaml
detection:
  row_threshold: 50
```

Add `row_threshold: int = 50` to `DetectionConfig` in TICKET-003.

## Edge Cases
- Single box: returns `[box]` unchanged
- All boxes in one row (e.g., two side-by-side panels): sorts right-to-left correctly
- Overlapping boxes (rare, from detection errors): treated as same row if y_centers are close

## Acceptance Criteria
- Given boxes at positions `[(x=100,y=50), (x=300,y=55), (x=200,y=300)]`, returns `[(x=300,y=55), (x=100,y=50), (x=200,y=300)]` (first row right-to-left, then second row)
- Empty input returns empty list
- Single-element input returns the same element

## Dependencies
- TICKET-001 (project skeleton)
- TICKET-002 (BoundingBox type)

## Estimated Effort
2 hours (including edge case testing)
