# TICKET-026: Pipeline Quality Benchmark vs manga-image-translator

## Summary
Run both pipelines on the same set of pages and produce a structured side-by-side comparison across detection, inpainting, OCR, and translation quality. The goal is to identify where manga-image-translator produces materially better output so those components can be swapped in or their techniques adopted.

## When to Run This Ticket
After TICKET-022 (batch chapter processor) is complete. The full pipeline must be functional before a meaningful end-to-end comparison is possible. Individual component benchmarks (detector, lama variant, OCR model) are done within their own tickets (005, 008, 010); this ticket covers holistic end-to-end output quality.

## Setup: Install manga-image-translator

Install into a separate venv to avoid dependency conflicts:
```bash
git clone https://github.com/zyddnys/manga-image-translator /tmp/mit
cd /tmp/mit
python -m venv .venv-mit
source .venv-mit/bin/activate
pip install -e .
python -m manga_translator --help
```

Run their pipeline on the benchmark page set:
```bash
python -m manga_translator \
  --mode folder \
  --detector ctd \
  --ocr 48px \
  --translator sugoi \
  --inpainter lama_large \
  --input tests/fixtures/benchmark/ \
  --dest /tmp/mit_output/
```

Use `sugoi` as their translator — it is the best offline Japanese-to-English option in their stack and the most direct comparison to our local Qwen2.5-7B.

## Benchmark Page Set

Create `tests/fixtures/benchmark/` with 10 pages covering:
- 2 pages: dense dialogue, multiple small bubbles
- 2 pages: pages with SFX mixed in with dialogue bubbles
- 2 pages: pages with handwritten/stylized character speech
- 2 pages: pages with minimal dialogue (2-3 bubbles, lots of art)
- 2 pages: pages with internal monologue (thought bubbles, small text)

Manually transcribe the Japanese text on all 10 pages to create a ground-truth file (`tests/fixtures/benchmark/ground_truth.json`).

## Evaluation Dimensions

### 1. Detection Quality
For each page, manually count:
- True positives (correct bubbles detected)
- False positives (panel borders, SFX, art elements mis-detected)
- False negatives (missed bubbles)

Record precision and recall for both pipelines. Expected: CTD outperforms YOLOv8n on FP rate; our bubble-boundary approach should have higher TP area coverage.

### 2. Inpainting Quality
Blind visual rating (1-5) by examining the inpainted canvas (before typesetting) on all 10 pages. Rate:
- Screentone continuity (does the dot pattern continue through the erased region?)
- Edge sharpness (is there a visible halo or box artifact around the erased area?)
- Background reconstruction (do structural lines like panel borders survive?)

### 3. OCR Accuracy
Compare detected Japanese text against `ground_truth.json` for both pipelines. Compute character error rate (CER). Expected: 48px_ctc may outperform manga-ocr on stylized crops; manga-ocr may win on standard vertical dialogue.

### 4. Translation Quality
This is the dimension where we should win. Rate 20 randomly selected dialogue lines (2 per page) on:
- **Accuracy**: Does the English meaning match the Japanese?
- **Register**: Does the translated line sound like the character described in `characters.json`?
- **Naturalness**: Does it read like something a native speaker would say?

Rate on a 1-3 scale per dimension per line. manga-image-translator has no character context injection, so their register scores should be lower on character-heavy pages.

Score each line independently, then average. Document any lines where they clearly win (complex sentence structures, idioms) vs. where we win (character register, pronoun consistency).

## Output Format

Write results to `tests/fixtures/benchmark/results.md`:

```markdown
# Benchmark Results — YYYY-MM-DD

## Detection
| Pipeline | Precision | Recall | Avg FP/page |
|---|---|---|---|
| manga-scanner | ... | ... | ... |
| MIT | ... | ... | ... |

## Inpainting (avg visual rating 1-5)
| Pipeline | Screentone | Edges | Background |
|---|---|---|---|
| manga-scanner | ... | ... | ... |
| MIT | ... | ... | ... |

## OCR (Character Error Rate, lower is better)
| Pipeline | CER | Empty rate |
|---|---|---|
| manga-scanner | ... | ... |
| MIT | ... | ... |

## Translation (avg rating 1-3)
| Pipeline | Accuracy | Register | Naturalness |
|---|---|---|---|
| manga-scanner | ... | ... | ... |
| MIT | ... | ... | ... |

## Action items from results
[concrete ticket amendments or new tickets]
```

## Expected Action Items

Based on the comparison summary written in TICKET-000:
- If CTD outperforms YOLOv8 on detection: create TICKET-027 to integrate CTD as the default detector
- If lama_large materially beats lama on visual quality: update TICKET-003 config default to `lama_large`
- If 48px_ctc CER is < manga-ocr CER by > 5%: create TICKET-028 to integrate 48px_ctc as an optional engine
- If MIT translation quality is competitive despite no character profiles: revisit our prompt design in TICKET-014

## Acceptance Criteria
- Both pipelines have been run on all 10 benchmark pages
- `results.md` is filled in with all four evaluation dimension tables
- At least one concrete action item is documented based on the results

## Dependencies
- TICKET-022 (our full pipeline must be complete)
- manga-image-translator installed in a separate environment

## Estimated Effort
6 hours (setup, runs, manual evaluation across 10 pages, writeup)
