# TICKET-015: Translation Wrapper, Response Parser, and Retry Logic

## Summary
Write the `Translator` class that calls the Ollama HTTP API, parses the JSON array response, validates it, and retries on failure. Include a circuit breaker for Ollama connectivity issues and a fallback that prevents the entire pipeline from crashing on a bad page.

## Language and Tools
- Python 3.11
- `httpx` (installed in TICKET-012) — sync HTTP client
- No additional packages

## Implementation

File: `src/manga_scanner/translation/translator.py`

```python
import json
import logging
import re
import httpx
from manga_scanner.config import TranslationConfig
from manga_scanner.translation.models import CharacterProfile
from manga_scanner.translation.prompt_builder import build_prompt
from manga_scanner.types import TranslationResult

logger = logging.getLogger(__name__)

_JSON_ARRAY_RE = re.compile(r'\[.*?\]', re.DOTALL)


class Translator:
    def __init__(self, config: TranslationConfig):
        self.config = config
        self.client = httpx.Client(
            base_url=config.ollama_url,
            timeout=config.timeout_seconds,
        )

    def translate_page(
        self,
        jp_texts: list[str],
        character_profiles: list[CharacterProfile],
        page_number: int = 0,
    ) -> TranslationResult:
        if not jp_texts:
            return TranslationResult(translations=[], raw_response="")

        system_prompt, user_prompt = build_prompt(jp_texts, character_profiles, page_number)
        raw_response = ""

        for attempt in range(self.config.max_retries + 1):
            try:
                raw_response = self._call_ollama(system_prompt, user_prompt)
            except httpx.ConnectError as e:
                # Ollama is not running — do not retry, fail fast
                logger.error("Ollama unreachable at %s: %s", self.config.ollama_url, e)
                return self._fallback(jp_texts, raw_response)
            except httpx.TimeoutException:
                logger.warning("Ollama timeout on attempt %d/%d", attempt + 1, self.config.max_retries + 1)
                if attempt == self.config.max_retries:
                    return self._fallback(jp_texts, raw_response)
                continue

            translations = self._parse_response(raw_response, len(jp_texts))
            if translations is not None:
                return TranslationResult(translations=translations, raw_response=raw_response)

            logger.warning(
                "Attempt %d/%d: could not parse valid JSON array from response. Raw: %.200s",
                attempt + 1, self.config.max_retries + 1, raw_response
            )

        logger.error("All retries exhausted for page %d. Using fallback.", page_number)
        return self._fallback(jp_texts, raw_response)

    def _call_ollama(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.config.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": self.config.temperature},
        }
        response = self.client.post("/api/chat", json=payload)
        response.raise_for_status()
        return response.json()["message"]["content"]

    def _parse_response(self, raw: str, expected_count: int) -> list[str] | None:
        # Try direct parse first
        try:
            result = json.loads(raw.strip())
            if isinstance(result, list) and len(result) == expected_count:
                return [str(s) for s in result]
        except json.JSONDecodeError:
            pass

        # Try extracting a JSON array from within surrounding text
        match = _JSON_ARRAY_RE.search(raw)
        if match:
            try:
                result = json.loads(match.group())
                if isinstance(result, list) and len(result) == expected_count:
                    return [str(s) for s in result]
            except json.JSONDecodeError:
                pass

        return None

    def _fallback(self, jp_texts: list[str], raw_response: str) -> TranslationResult:
        """
        Last resort: return the original Japanese text as the translation.
        The page will render with untranslated text rather than crashing.
        """
        logger.warning("Using fallback: returning original Japanese text for %d bubbles.", len(jp_texts))
        return TranslationResult(translations=list(jp_texts), raw_response=raw_response)

    def close(self):
        self.client.close()
```

## Retry Behavior

| Failure Mode | Behavior |
|---|---|
| JSON parse fails, array wrong length | Retry up to `max_retries` times |
| Ollama timeout | Retry up to `max_retries` times |
| `httpx.ConnectError` (Ollama down) | Circuit break immediately, no retries |
| All retries exhausted | Return fallback (original JP text) |

The fallback returns the Japanese text, not empty strings. This means the typesetter renders Japanese characters over the inpainted bubble — still readable by a Japanese speaker and clearly marks the failure for review.

## Regex Extraction Note
The `_JSON_ARRAY_RE` pattern handles the common case where the model outputs the array with surrounding explanation text despite being instructed not to. It extracts the first `[...]` block. If the model returns multiple arrays, only the first is used.

## Acceptance Criteria
- Given a mocked Ollama response of `'["Hello", "Wait"]'`, `_parse_response` returns `["Hello", "Wait"]`
- Given a response wrapped in prose: `'Here is the translation: ["Hello"]\nHope that helps'`, regex extraction succeeds
- Given a response with wrong array length, returns `None`
- `ConnectError` triggers immediate `_fallback` without retrying
- Fallback returns a `TranslationResult` with `translations` equal to the input `jp_texts`

## Dependencies
- TICKET-001 (project skeleton)
- TICKET-002 (TranslationResult type)
- TICKET-003 (TranslationConfig)
- TICKET-012 (Ollama running, httpx installed)
- TICKET-013 (CharacterProfile)
- TICKET-014 (build_prompt)

## Estimated Effort
3 hours
