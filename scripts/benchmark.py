"""TICKET-026: Pipeline quality benchmark — manga-scanner vs manga-image-translator.

Workflow
--------
1. Add real manga pages to tests/fixtures/benchmark/ and fill in ground_truth.json.
2. Run our pipeline with --dump-dir to get per-page JSON:
       manga-scan chapter -i tests/fixtures/benchmark/ -o /tmp/our_output/ \\
           --dump-dir /tmp/our_dumps/
3. Run MIT pipeline (see scripts/run_mit.sh) and point --mit-ocr-dir at their
   output directory where *.txt files were saved (one bubble per line per file).
4. Run this script:
       python scripts/benchmark.py \\
           --ground-truth tests/fixtures/benchmark/ground_truth.json \\
           --our-dump-dir /tmp/our_dumps/ \\
           --mit-ocr-dir /tmp/mit_output/ \\
           --output tests/fixtures/benchmark/results.md

Sections that require manual input (inpainting visual ratings, translation quality
scores, detection TP/FP/FN counts) are left as blank rows in the output template.

MIT OCR text format
-------------------
manga-image-translator saves recognised text when run with --save-text. It writes
one .txt file per input image, with each recognised text region on its own line
in reading order. Place those files in --mit-ocr-dir with the same stems as the
input images (page_001.txt, page_002.txt, ...).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path


# ---------------------------------------------------------------------------
# CER (Character Error Rate)
# ---------------------------------------------------------------------------

def _levenshtein(a: str, b: str) -> int:
    """Standard dynamic-programming Levenshtein distance."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def cer(hypothesis: str, reference: str) -> float:
    """Character Error Rate: edit_distance / len(reference). 0.0 = perfect."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return _levenshtein(hypothesis, reference) / len(reference)


def corpus_cer(hypotheses: list[str], references: list[str]) -> float:
    """CER over a list of (hypothesis, reference) pairs, weighted by reference length."""
    total_errors = sum(_levenshtein(h, r) for h, r in zip(hypotheses, references))
    total_chars = sum(len(r) for r in references)
    return total_errors / total_chars if total_chars else 0.0


# ---------------------------------------------------------------------------
# Detection metrics (IoU-based matching)
# ---------------------------------------------------------------------------

def iou(box_a: list[int], box_b: list[int]) -> float:
    """Intersection over union for two boxes [x1, y1, x2, y2]."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x = max(0, min(xa2, xb2) - max(xa1, xb1))
    inter_y = max(0, min(ya2, yb2) - max(ya1, yb1))
    inter = inter_x * inter_y
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class DetectionCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0


def detection_metrics(
    pred_boxes: list[list[int]],
    gt_boxes: list[list[int]],
    iou_threshold: float = 0.5,
) -> DetectionCounts:
    """Match predicted boxes to ground-truth boxes greedily by descending IoU."""
    matched_gt: set[int] = set()
    counts = DetectionCounts()
    for pred in pred_boxes:
        best_iou = 0.0
        best_gt_idx = -1
        for j, gt in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            score = iou(pred, gt)
            if score > best_iou:
                best_iou = score
                best_gt_idx = j
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            counts.tp += 1
            matched_gt.add(best_gt_idx)
        else:
            counts.fp += 1
    counts.fn = len(gt_boxes) - len(matched_gt)
    return counts


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ground_truth(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_our_dumps(dump_dir: Path) -> dict[str, dict]:
    """Load per-page JSON files produced by process_chapter(dump_dir=...)."""
    result: dict[str, dict] = {}
    for p in sorted(dump_dir.glob("*.json")):
        with p.open(encoding="utf-8") as f:
            data = json.load(f)
        result[data["filename"]] = data
    return result


def load_mit_ocr(mit_dir: Path) -> dict[str, list[str]]:
    """Load MIT OCR text files (one bubble per line, stem matches input image)."""
    result: dict[str, list[str]] = {}
    for p in sorted(mit_dir.glob("*.txt")):
        lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        result[p.stem] = lines
    return result


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

@dataclass
class OcrPageMetrics:
    filename: str
    our_cer: float
    mit_cer: float | None
    our_empty_rate: float
    mit_empty_rate: float | None
    n_gt_bubbles: int


@dataclass
class DetectionPageMetrics:
    filename: str
    our: DetectionCounts
    n_gt_bubbles: int


def compute_ocr_metrics(
    ground_truth: dict,
    our_dumps: dict[str, dict],
    mit_ocr: dict[str, list[str]],
) -> list[OcrPageMetrics]:
    metrics = []
    for page in ground_truth["pages"]:
        filename = page["filename"]
        stem = Path(filename).stem
        gt_texts = [b["japanese"] for b in page["bubbles"] if b["japanese"] != "PLACEHOLDER — transcribe actual text here" and not b["japanese"].startswith("PLACEHOLDER")]

        if not gt_texts:
            continue

        # Our pipeline
        our_texts: list[str] = []
        if filename in our_dumps:
            our_texts = [r["text"] for r in our_dumps[filename].get("ocr", [])]

        if our_texts:
            paired = list(zip(our_texts[: len(gt_texts)], gt_texts[: len(our_texts)]))
            our_cer_val = corpus_cer([h for h, _ in paired], [r for _, r in paired])
            our_empty = sum(1 for t in our_texts if not t.strip()) / len(our_texts)
        else:
            our_cer_val = 1.0
            our_empty = 1.0

        # MIT
        mit_texts = mit_ocr.get(stem)
        if mit_texts is not None:
            paired_m = list(zip(mit_texts[: len(gt_texts)], gt_texts[: len(mit_texts)]))
            mit_cer_val: float | None = corpus_cer([h for h, _ in paired_m], [r for _, r in paired_m])
            mit_empty: float | None = sum(1 for t in mit_texts if not t.strip()) / len(mit_texts)
        else:
            mit_cer_val = None
            mit_empty = None

        metrics.append(OcrPageMetrics(
            filename=filename,
            our_cer=our_cer_val,
            mit_cer=mit_cer_val,
            our_empty_rate=our_empty,
            mit_empty_rate=mit_empty,
            n_gt_bubbles=len(gt_texts),
        ))
    return metrics


def compute_detection_metrics_all(
    ground_truth: dict,
    our_dumps: dict[str, dict],
    iou_threshold: float = 0.5,
) -> list[DetectionPageMetrics]:
    metrics = []
    for page in ground_truth["pages"]:
        filename = page["filename"]
        gt_boxes = [b["bbox"] for b in page["bubbles"] if b["bbox"] != [0, 0, 0, 0]]
        if not gt_boxes:
            continue
        our_boxes: list[list[int]] = []
        if filename in our_dumps:
            our_boxes = [b["bbox"] for b in our_dumps[filename].get("boxes", [])]
        counts = detection_metrics(our_boxes, gt_boxes, iou_threshold)
        metrics.append(DetectionPageMetrics(
            filename=filename,
            our=counts,
            n_gt_bubbles=len(gt_boxes),
        ))
    return metrics


# ---------------------------------------------------------------------------
# Results.md generation
# ---------------------------------------------------------------------------

def _pct(v: float | None) -> str:
    if v is None:
        return "N/A"
    return f"{v:.1%}"


def write_results_md(
    ocr_metrics: list[OcrPageMetrics],
    det_metrics: list[DetectionPageMetrics],
    output_path: Path,
) -> None:
    today = date.today().isoformat()

    # Detection aggregates
    if det_metrics:
        total_tp = sum(m.our.tp for m in det_metrics)
        total_fp = sum(m.our.fp for m in det_metrics)
        total_fn = sum(m.our.fn for m in det_metrics)
        agg_our = DetectionCounts(tp=total_tp, fp=total_fp, fn=total_fn)
        avg_fp = total_fp / len(det_metrics)
    else:
        agg_our = DetectionCounts()
        avg_fp = 0.0

    # OCR aggregates
    our_cers = [m.our_cer for m in ocr_metrics]
    mit_cers = [m.mit_cer for m in ocr_metrics if m.mit_cer is not None]
    our_empties = [m.our_empty_rate for m in ocr_metrics]
    mit_empties = [m.mit_empty_rate for m in ocr_metrics if m.mit_empty_rate is not None]
    avg_our_cer = sum(our_cers) / len(our_cers) if our_cers else None
    avg_mit_cer = sum(mit_cers) / len(mit_cers) if mit_cers else None
    avg_our_empty = sum(our_empties) / len(our_empties) if our_empties else None
    avg_mit_empty = sum(mit_empties) / len(mit_empties) if mit_empties else None

    lines = [
        f"# Benchmark Results — {today}",
        "",
        "## Detection",
        "",
        "| Pipeline | Precision | Recall | Avg FP/page |",
        "|---|---|---|---|",
        f"| manga-scanner | {_pct(agg_our.precision)} | {_pct(agg_our.recall)} | {avg_fp:.1f} |",
        "| MIT | MANUAL | MANUAL | MANUAL |",
        "",
        "> Detection counts for MIT require manually comparing their output images against",
        "> ground_truth.json bounding boxes. Update the row above after manual review.",
        "",
        "## Inpainting (avg visual rating 1–5)",
        "",
        "| Pipeline | Screentone | Edges | Background |",
        "|---|---|---|---|",
        "| manga-scanner | — | — | — |",
        "| MIT | — | — | — |",
        "",
        "> Rate each page independently on screentone continuity (dot pattern survives erase),",
        "> edge sharpness (no visible halo/box artifact), and background reconstruction",
        "> (panel borders intact). Enter per-page ratings in the table below, then average.",
        "",
        "### Per-page inpainting ratings",
        "",
        "| Page | Our screentone | Our edges | Our bg | MIT screentone | MIT edges | MIT bg |",
        "|---|---|---|---|---|---|---|",
    ]
    for page in (det_metrics if det_metrics else []):
        lines.append(f"| {page.filename} | — | — | — | — | — | — |")
    lines += [
        "",
        "## OCR (Character Error Rate, lower is better)",
        "",
        "| Pipeline | CER | Empty rate |",
        "|---|---|---|",
        f"| manga-scanner | {_pct(avg_our_cer)} | {_pct(avg_our_empty)} |",
        f"| MIT | {_pct(avg_mit_cer)} | {_pct(avg_mit_empty)} |",
        "",
        "### Per-page OCR breakdown",
        "",
        "| Page | GT bubbles | Our CER | MIT CER |",
        "|---|---|---|---|",
    ]
    for m in ocr_metrics:
        lines.append(f"| {m.filename} | {m.n_gt_bubbles} | {_pct(m.our_cer)} | {_pct(m.mit_cer)} |")
    lines += [
        "",
        "## Translation (avg rating 1–3)",
        "",
        "| Pipeline | Accuracy | Register | Naturalness |",
        "|---|---|---|---|",
        "| manga-scanner | — | — | — |",
        "| MIT | — | — | — |",
        "",
        "> Select 2 dialogue lines per page (20 total). Rate each 1-3:",
        "> Accuracy = does English meaning match Japanese?",
        "> Register = does it sound like the character profile?",
        "> Naturalness = would a native speaker say this?",
        "",
        "### Per-line translation ratings",
        "",
        "| Page | Bubble | Japanese | Our EN | MIT EN | Our Acc | Our Reg | Our Nat | MIT Acc | MIT Reg | MIT Nat |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
        "| | | PLACEHOLDER | | | — | — | — | — | — | — |",
        "",
        "## Action items from results",
        "",
        "- [ ] If CTD outperforms YOLOv8 on detection precision: open TICKET-027 (CTD integration)",
        "- [ ] If lama_large beats lama visually: update config.yaml default to lama_large",
        "- [ ] If MIT OCR CER < ours by >5%: open TICKET-028 (48px_ctc engine option)",
        "- [ ] If MIT translation competitive despite no character profiles: revisit TICKET-014 prompt design",
        "",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute benchmark metrics for TICKET-026.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ground-truth",
        required=True,
        type=Path,
        help="Path to ground_truth.json.",
    )
    parser.add_argument(
        "--our-dump-dir",
        type=Path,
        default=None,
        help="Directory of per-page JSON files from manga-scan chapter --dump-dir.",
    )
    parser.add_argument(
        "--mit-ocr-dir",
        type=Path,
        default=None,
        help="Directory of .txt files from MIT pipeline --save-text.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/benchmark/results.md"),
        help="Where to write results.md (default: tests/fixtures/benchmark/results.md).",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for detection TP matching (default: 0.5).",
    )
    args = parser.parse_args()

    gt = load_ground_truth(args.ground_truth)

    our_dumps = load_our_dumps(args.our_dump_dir) if args.our_dump_dir else {}
    mit_ocr = load_mit_ocr(args.mit_ocr_dir) if args.mit_ocr_dir else {}

    ocr_m = compute_ocr_metrics(gt, our_dumps, mit_ocr)
    det_m = compute_detection_metrics_all(gt, our_dumps, args.iou_threshold)

    if not ocr_m and not det_m:
        print(
            "No comparable data found. Make sure ground_truth.json has real transcriptions "
            "and --our-dump-dir points to JSON files from manga-scan chapter --dump-dir.",
            file=sys.stderr,
        )

    write_results_md(ocr_m, det_m, args.output)


if __name__ == "__main__":
    main()
