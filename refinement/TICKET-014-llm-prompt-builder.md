# TICKET-014: LLM Prompt Builder

## Summary
Write the function that constructs the system and user prompts sent to Qwen2.5-7B-Instruct via Ollama. The prompt must inject character context, instruct structured JSON output, and present Japanese text in numbered reading order — all within a token budget that leaves room for the model to generate translations.

## Language and Tools
- Python 3.11 standard library only
- No additional packages

## Token Budget Context

Qwen2.5-7B-Instruct has a 32K token context window. A typical manga page has 5-20 speech bubbles. Budget allocation:
- System prompt: ~150 tokens
- Character profiles: ~100 tokens per character (keep the list short)
- Page content (Japanese text): ~10-50 tokens per bubble
- Model output (English translations): ~10-50 tokens per bubble

For a 20-bubble page with 3 characters: ~150 + 300 + 1000 + 1000 = ~2450 tokens. Well within limits. No chunking needed.

## Implementation

File: `src/manga_scanner/translation/prompt_builder.py`

```python
import json
from manga_scanner.translation.models import CharacterProfile, SpeechRegister


SYSTEM_PROMPT = """You are a professional manga localizer. Translate the numbered Japanese dialogue into natural English.

Rules:
- Return ONLY a valid JSON array of translated strings, one per input line.
- Preserve speech register and personality for each character as described.
- Do not add notes, explanations, or any text outside the JSON array.
- If a line is sound effect text (SFX), transliterate or adapt it naturally.
- Output array must have exactly the same number of elements as the input array."""


def _format_character_block(profiles: list[CharacterProfile]) -> str:
    if not profiles:
        return ""
    lines = ["Characters in this scene:"]
    for p in profiles:
        rel_str = ""
        if p.relationships:
            pairs = [f"{p.name} sees {other} as {role}" for other, role in p.relationships.items()]
            rel_str = " " + ". ".join(pairs) + "."
        lines.append(
            f"- {p.name} ({p.jp_name}): {p.speech_register.value} register. "
            f"{p.pronouns} pronouns. {p.speech_notes}{rel_str}"
        )
    return "\n".join(lines)


def build_prompt(
    jp_texts: list[str],
    character_profiles: list[CharacterProfile],
    page_number: int = 0,
) -> tuple[str, str]:
    """
    Returns (system_prompt, user_prompt) ready for the Ollama chat API.
    jp_texts must already be in reading order (output of sort_reading_order + OCR).
    """
    char_block = _format_character_block(character_profiles)

    numbered_lines = "\n".join(
        f"{i + 1}. {text}" for i, text in enumerate(jp_texts)
    )

    user_parts = []
    if char_block:
        user_parts.append(char_block)
    user_parts.append(f"Page {page_number} dialogue to translate:")
    user_parts.append(numbered_lines)
    user_parts.append(
        f'\nReturn a JSON array with exactly {len(jp_texts)} translated strings.'
    )

    return SYSTEM_PROMPT, "\n\n".join(user_parts)
```

## Example Output

For a page with 3 bubbles and no character profiles:
```
System: [SYSTEM_PROMPT as above]

User:
Page 1 dialogue to translate:
1. 行くぞ！
2. 待ってください！
3. もう遅い。

Return a JSON array with exactly 3 translated strings.
```

Expected model response:
```json
["Let's go!", "Please wait!", "It's too late."]
```

## Design Decisions

- Temperature should be 0.2 (set in TICKET-003 config). Lower temperature = more deterministic, more literal translation. Increase to 0.4 if translations feel too stiff.
- The system prompt explicitly prohibits any text outside the JSON array. This is the primary guard against malformed output that triggers the retry logic in TICKET-015.
- Character `speech_notes` is injected verbatim. Keep them to one or two sentences to avoid drowning the translation content in context.
- `page_number` is included in the prompt as lightweight context. The LLM does not use it for translation logic but it helps if you're debugging prompt/response pairs.

## Acceptance Criteria
- `build_prompt(["行くぞ！"], [], page_number=1)` returns a tuple of two non-empty strings
- The user prompt contains the numbered Japanese text
- The user prompt contains the exact count constraint: "exactly N translated strings"
- `build_prompt(jp_texts, profiles)` with profiles produces a character block section
- `build_prompt([], [])` returns valid strings (edge case: empty page — orchestrator should skip this before calling the builder)

## Dependencies
- TICKET-001 (project skeleton)
- TICKET-013 (CharacterProfile model)

## Estimated Effort
2 hours
