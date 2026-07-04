# TICKET-013: Character Profile JSON Schema and Pydantic Model

## Summary
Define the `characters.json` file format and corresponding Pydantic model that carries per-character metadata into the translation prompt. Character profiles let the LLM preserve speech register (formal, rough, childlike), consistent pronoun choices, and relationship-aware phrasing across a chapter.

## Language and Tools
- Python 3.11
- `pydantic` v2 (already installed as pydantic-settings dependency from TICKET-003)

## File: `characters.json`

This file lives in the project root and is user-maintained per manga series.

```json
{
  "characters": [
    {
      "name": "Kira",
      "jp_name": "キラ",
      "speech_register": "formal",
      "pronouns": "he/him",
      "speech_notes": "Speaks in long sentences with deliberate word choices. Never uses contractions. Refers to himself by name occasionally.",
      "relationships": {
        "Ryuk": "tool",
        "L": "nemesis"
      }
    },
    {
      "name": "Ryuk",
      "jp_name": "リューク",
      "speech_register": "casual",
      "pronouns": "he/him",
      "speech_notes": "Dry, amused tone. Short sentences. Frequently makes observations about humans as curiosities.",
      "relationships": {
        "Kira": "owner"
      }
    }
  ]
}
```

## Pydantic Model

File: `src/manga_scanner/translation/models.py`

```python
from enum import Enum
from pathlib import Path
from pydantic import BaseModel
import json


class SpeechRegister(str, Enum):
    FORMAL = "formal"
    CASUAL = "casual"
    ROUGH = "rough"
    ARCHAIC = "archaic"
    CHILDLIKE = "childlike"


class CharacterProfile(BaseModel):
    name: str
    jp_name: str = ""
    speech_register: SpeechRegister = SpeechRegister.CASUAL
    pronouns: str = "they/them"
    speech_notes: str = ""
    relationships: dict[str, str] = {}


class CharacterDatabase(BaseModel):
    characters: list[CharacterProfile]


def load_character_profiles(path: Path) -> list[CharacterProfile]:
    """Returns empty list if file does not exist — pipeline runs without profiles."""
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return CharacterDatabase(**data).characters
```

## Usage Contract

- `characters.json` is optional. If it does not exist, `load_character_profiles()` returns `[]` and the prompt builder (TICKET-014) omits the character context block.
- The `speech_notes` field is the most important. The LLM uses it directly as a translation instruction. Keep notes concise (one to two sentences) — the system prompt has limited token budget.
- `relationships` is informational context, not directly injected into the prompt as a structured object. The prompt builder serializes it as a readable sentence: "Kira views Ryuk as a tool."

## Acceptance Criteria
- `load_character_profiles(Path("characters.json"))` returns a list of `CharacterProfile` objects with correct enum values
- Invalid `speech_register` value in JSON raises a Pydantic `ValidationError`
- `load_character_profiles(Path("nonexistent.json"))` returns `[]` without raising
- `CharacterDatabase` validates the full nested structure

## Dependencies
- TICKET-001 (project skeleton)
- TICKET-003 (pydantic already installed)

## Estimated Effort
1.5 hours
