from __future__ import annotations

from manga_scanner.translation.models import CharacterProfile


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
    jp_texts must already be in reading order.
    """
    char_block = _format_character_block(character_profiles)

    numbered_lines = "\n".join(
        f"{i + 1}. {text}" for i, text in enumerate(jp_texts)
    )

    user_parts: list[str] = []
    if char_block:
        user_parts.append(char_block)
    user_parts.append(f"Page {page_number} dialogue to translate:")
    user_parts.append(numbered_lines)
    user_parts.append(
        f"\nReturn a JSON array with exactly {len(jp_texts)} translated strings."
    )

    return SYSTEM_PROMPT, "\n\n".join(user_parts)
