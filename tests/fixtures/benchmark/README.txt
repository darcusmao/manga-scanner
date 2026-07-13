Benchmark images are NOT committed to this repo.

Manga pages are copyrighted by their publishers (Shueisha, Kodansha, etc.).
Committing them to a public repository risks DMCA takedowns.

To run the benchmark locally:
1. Copy 10 manga page scans into this directory as page_001.png ... page_010.png
   (or any .png/.jpg filenames — the pipeline accepts any image files)
2. Fill in ground_truth.json with the Japanese text transcriptions for each page
3. Follow the instructions in scripts/benchmark.py to run both pipelines and
   generate results.md

Suggested page categories (2 pages each):
  - dense_dialogue   multiple small speech bubbles per panel
  - sfx_mixed        SFX text mixed in with dialogue bubbles
  - stylized         handwritten or decorative character speech
  - minimal          2-3 bubbles, lots of art
  - monologue        thought bubbles / small internal-monologue text

ground_truth.json and results.md are also gitignored after they contain real
transcriptions, since those are derivative works of the original manga.
