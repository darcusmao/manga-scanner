# TICKET-012: Ollama Installation and Qwen2.5-7B Model Pull

## Summary
Install Ollama, pull the quantized Qwen2.5-7B-Instruct model, verify the server is reachable, and document the startup procedure. Ollama runs as a background daemon outside the Python process — this ticket covers all setup steps before the Python translator is written in TICKET-015.

## What Ollama Is
Ollama is a local LLM inference server. It:
- Runs as a background process on port 11434
- Manages model loading/unloading from VRAM automatically
- Exposes an OpenAI-compatible REST API
- Handles GGUF model files natively (no Python-side llama.cpp bindings needed)

This is the preferred approach over `llama.cpp` Python bindings because it decouples the LLM process from the Python pipeline process, preventing VRAM conflicts with PyTorch-managed models.

## Installation

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

After installation, start the daemon:
```bash
ollama serve
```
Leave this running in a terminal or configure it as a system service. On macOS, `brew services start ollama` will start it at login.

## Model Selection and Pull

Model: `qwen2.5:7b-instruct-q4_K_M`

Why this model:
- Qwen2.5-7B-Instruct: strong instruction-following, trained with multilingual data including Japanese-to-English translation tasks
- Q4_K_M quantization: 4-bit with K-quants variant M — best quality/size tradeoff for 7B parameter models; negligible quality loss vs. FP16
- VRAM usage: ~5.0 GB at Q4_K_M
- Download size: ~4.7 GB

Pull the model:
```bash
ollama pull qwen2.5:7b-instruct-q4_K_M
```

This is a one-time download. The model is stored in `~/.ollama/models/`.

## Verification

After `ollama serve` is running and the model is pulled:

```bash
curl -s http://localhost:11434/api/chat -d '{
  "model": "qwen2.5:7b-instruct-q4_K_M",
  "messages": [{"role": "user", "content": "Reply with only the word: ready"}],
  "stream": false
}' | python3 -c "import sys, json; print(json.load(sys.stdin)['message']['content'])"
```

Expected output: `ready` (or similar single-word acknowledgment)

## Python HTTP Client
Install `httpx` for making HTTP calls to Ollama from the pipeline:
```bash
uv add httpx
```

`httpx` is preferred over `requests` because it supports both sync and async, and has cleaner timeout handling.

## VRAM Interaction Note
Ollama loads the model to VRAM on the first request and keeps it there for a configurable idle period (default: 5 minutes). This means:
- Before calling the translator, ensure iopaint and manga-ocr have been unloaded (TICKET-020)
- Ollama cannot be force-unloaded from Python; it will evict the model from VRAM naturally after the idle timeout
- On an 8GB GPU: Ollama's 5GB + any lingering PyTorch allocations will cause OOM. The VRAM lifecycle in TICKET-020 is mandatory.

## Acceptance Criteria
- `ollama list` shows `qwen2.5:7b-instruct-q4_K_M` in the model list
- The verification curl command returns a coherent response
- `uv run python -c "import httpx; httpx.get('http://localhost:11434/')"` returns HTTP 200

## Dependencies
- TICKET-004 (CUDA confirmed — Ollama uses the same GPU)

## Estimated Effort
2 hours (including download time)
