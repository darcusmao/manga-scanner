from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod

import httpx

from manga_scanner.config import TranslationConfig
from manga_scanner.translation.models import CharacterProfile
from manga_scanner.translation.prompt_builder import build_prompt
from manga_scanner.types import TranslationResult

logger = logging.getLogger(__name__)

_JSON_ARRAY_RE = re.compile(r'\[.*?\]', re.DOTALL)


class BaseTranslator(ABC):
    @abstractmethod
    def translate_page(
        self,
        jp_texts: list[str],
        character_profiles: list[CharacterProfile],
        page_number: int = 0,
    ) -> TranslationResult:
        ...

    def close(self) -> None:
        pass

    def _fallback(self, jp_texts: list[str], raw_response: str = "") -> TranslationResult:
        logger.warning(
            "Using fallback: returning original Japanese text for %d bubbles.",
            len(jp_texts),
        )
        return TranslationResult(translations=list(jp_texts), raw_response=raw_response)


class OllamaTranslator(BaseTranslator):
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

    def close(self) -> None:
        self.client.close()


class DeepLTranslator(BaseTranslator):
    """
    Uses the DeepL REST API. Free-tier keys end in ':fx' and hit api-free.deepl.com.
    Paid keys hit api.deepl.com. Set key via MANGA_TRANSLATION__DEEPL_API_KEY env var.
    """

    def __init__(self, config: TranslationConfig) -> None:
        self.config = config
        base_url = (
            "https://api-free.deepl.com"
            if config.deepl_api_key.endswith(":fx")
            else "https://api.deepl.com"
        )
        self.client = httpx.Client(
            base_url=base_url,
            timeout=config.timeout_seconds,
            headers={"Authorization": f"DeepL-Auth-Key {config.deepl_api_key}"},
        )

    def translate_page(
        self,
        jp_texts: list[str],
        character_profiles: list[CharacterProfile],
        page_number: int = 0,
    ) -> TranslationResult:
        if not jp_texts:
            return TranslationResult(translations=[], raw_response="")
        try:
            response = self.client.post(
                "/v2/translate",
                json={
                    "text": jp_texts,
                    "source_lang": "JA",
                    "target_lang": "EN-US",
                },
            )
            response.raise_for_status()
            data = response.json()
            translations = [t["text"] for t in data["translations"]]
            return TranslationResult(translations=translations, raw_response=json.dumps(data))
        except httpx.HTTPStatusError as e:
            logger.error("DeepL API error %d: %s", e.response.status_code, e.response.text)
            return self._fallback(jp_texts)
        except Exception as e:
            logger.error("DeepL translation failed: %s", e)
            return self._fallback(jp_texts)

    def close(self) -> None:
        self.client.close()


class GoogleTranslator(BaseTranslator):
    """
    Uses the Google Cloud Translation REST API (Basic / v2).
    Set key via MANGA_TRANSLATION__GOOGLE_API_KEY env var.
    """

    def __init__(self, config: TranslationConfig) -> None:
        self.config = config
        self.client = httpx.Client(
            base_url="https://translation.googleapis.com",
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
        try:
            response = self.client.post(
                "/language/translate/v2",
                params={"key": self.config.google_api_key},
                json={
                    "q": jp_texts,
                    "source": "ja",
                    "target": "en",
                    "format": "text",
                },
            )
            response.raise_for_status()
            data = response.json()
            translations = [t["translatedText"] for t in data["data"]["translations"]]
            return TranslationResult(translations=translations, raw_response=json.dumps(data))
        except httpx.HTTPStatusError as e:
            logger.error("Google Translate API error %d: %s", e.response.status_code, e.response.text)
            return self._fallback(jp_texts)
        except Exception as e:
            logger.error("Google translation failed: %s", e)
            return self._fallback(jp_texts)

    def close(self) -> None:
        self.client.close()


# Backward-compat alias so existing test imports still work
Translator = OllamaTranslator


def create_translator(config: TranslationConfig) -> BaseTranslator:
    if config.backend == "deepl":
        if not config.deepl_api_key:
            raise ValueError("DeepL backend selected but deepl_api_key is not set. "
                             "Set MANGA_TRANSLATION__DEEPL_API_KEY env var.")
        return DeepLTranslator(config)
    if config.backend == "google":
        if not config.google_api_key:
            raise ValueError("Google backend selected but google_api_key is not set. "
                             "Set MANGA_TRANSLATION__GOOGLE_API_KEY env var.")
        return GoogleTranslator(config)
    return OllamaTranslator(config)
