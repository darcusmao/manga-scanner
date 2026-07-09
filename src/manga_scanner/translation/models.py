from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

from pydantic import BaseModel


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
