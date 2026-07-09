from __future__ import annotations

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
    def __init__(self, config: TranslationConfig) -> None:
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
                logger.error("Ollama unreachable at %s: %s", self.config.ollama_url, e)
                return self._fallback(jp_texts, raw_response)
            except httpx.TimeoutException:
                logger.warning(
                    "Ollama timeout on attempt %d/%d",
                    attempt + 1, self.config.max_retries + 1,
                )
                if attempt == self.config.max_retries:
                    return self._fallback(jp_texts, raw_response)
                continue

            translations = self._parse_response(raw_response, len(jp_texts))
            if translations is not None:
                return TranslationResult(translations=translations, raw_response=raw_response)

            logger.warning(
                "Attempt %d/%d: could not parse valid JSON array. Raw: %.200s",
                attempt + 1, self.config.max_retries + 1, raw_response,
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
        try:
            result = json.loads(raw.strip())
            if isinstance(result, list) and len(result) == expected_count:
                return [str(s) for s in result]
        except json.JSONDecodeError:
            pass

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
        logger.warning(
            "Using fallback: returning original Japanese text for %d bubbles.",
            len(jp_texts),
        )
        return TranslationResult(translations=list(jp_texts), raw_response=raw_response)

    def close(self) -> None:
        self.client.close()
