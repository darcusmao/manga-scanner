#!/usr/bin/env bash
# TICKET-026: Install manga-image-translator and run it on the benchmark pages.
# Run from the project root.
set -euo pipefail

BENCHMARK_DIR="tests/fixtures/benchmark"
MIT_SRC="/tmp/mit"
MIT_OUT="/tmp/mit_output"

# ---- Install ---------------------------------------------------------------
if [ ! -d "$MIT_SRC" ]; then
    git clone https://github.com/zyddnys/manga-image-translator "$MIT_SRC"
fi

if [ ! -d "$MIT_SRC/.venv-mit" ]; then
    python -m venv "$MIT_SRC/.venv-mit"
    source "$MIT_SRC/.venv-mit/bin/activate"
    pip install -e "$MIT_SRC"
else
    source "$MIT_SRC/.venv-mit/bin/activate"
fi

# ---- Run -------------------------------------------------------------------
mkdir -p "$MIT_OUT"

python -m manga_translator \
    --mode folder \
    --detector ctd \
    --ocr 48px \
    --translator sugoi \
    --inpainter lama_large \
    --save-text \
    --input "$BENCHMARK_DIR" \
    --dest "$MIT_OUT"

echo ""
echo "MIT output written to $MIT_OUT"
echo "Translated images:  $MIT_OUT/*.png"
echo "OCR text files:     $MIT_OUT/*.txt"
echo ""
echo "Next step: run the benchmark script:"
echo "  python scripts/benchmark.py \\"
echo "    --ground-truth $BENCHMARK_DIR/ground_truth.json \\"
echo "    --our-dump-dir /tmp/our_dumps/ \\"
echo "    --mit-ocr-dir $MIT_OUT \\"
echo "    --output $BENCHMARK_DIR/results.md"
