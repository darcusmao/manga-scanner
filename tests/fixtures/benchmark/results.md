# Benchmark Results — PENDING

> Run `python scripts/benchmark.py --help` to generate this file from pipeline outputs.
> Fill in manual sections (inpainting visual ratings, translation quality) by hand.

## Detection

| Pipeline | Precision | Recall | Avg FP/page |
|---|---|---|---|
| manga-scanner | — | — | — |
| MIT | — | — | — |

## Inpainting (avg visual rating 1–5)

| Pipeline | Screentone | Edges | Background |
|---|---|---|---|
| manga-scanner | — | — | — |
| MIT | — | — | — |

## OCR (Character Error Rate, lower is better)

| Pipeline | CER | Empty rate |
|---|---|---|
| manga-scanner | — | — |
| MIT | — | — |

## Translation (avg rating 1–3)

| Pipeline | Accuracy | Register | Naturalness |
|---|---|---|---|
| manga-scanner | — | — | — |
| MIT | — | — | — |

## Action items from results

- [ ] If CTD outperforms YOLOv8 on detection precision: open TICKET-027 (CTD integration)
- [ ] If lama_large beats lama visually: update config.yaml default to lama_large
- [ ] If MIT OCR CER < ours by >5%: open TICKET-028 (48px_ctc engine option)
- [ ] If MIT translation competitive despite no character profiles: revisit TICKET-014 prompt design
