"""Tests for TICKET-017 (fitter), TICKET-018 (renderer), and TICKET-013 (character profiles)."""
import sys
import pytest
from pathlib import Path
from PIL import Image

from manga_scanner.typesetting.fitter import fit_text, FitResult
from manga_scanner.detection.sorter import sort_reading_order
from manga_scanner.types import BoundingBox

if sys.platform == "darwin":
    TEST_FONT = "/System/Library/Fonts/Helvetica.ttc"
else:
    TEST_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def make_box(x1, y1, x2, y2) -> BoundingBox:
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=0.9)


class TestFitText:
    def test_short_text_uses_max_font_size(self):
        result = fit_text("Hi", 300, 100, TEST_FONT, max_font_size=24, min_font_size=8)
        assert result.font_size == 24
        assert result.fits is True

    def test_long_text_reduces_font_size(self):
        long_text = "This is a very long sentence that will not fit at a large font size."
        result = fit_text(long_text, 100, 60, TEST_FONT, max_font_size=24, min_font_size=8)
        assert result.font_size < 24

    def test_impossible_text_returns_min_size(self):
        very_long = "W " * 200
        result = fit_text(very_long, 50, 30, TEST_FONT, max_font_size=24, min_font_size=8)
        assert result.font_size == 8
        assert result.fits is False

    def test_empty_text_does_not_raise(self):
        result = fit_text("", 200, 100, TEST_FONT)
        assert isinstance(result, FitResult)
        assert result.lines == []

    def test_lines_reconstruct_original_words(self):
        text = "Hello world how are you"
        result = fit_text(text, 300, 200, TEST_FONT)
        reconstructed = " ".join(" ".join(result.lines).split())
        assert reconstructed == text

    def test_single_word_returns_one_line(self):
        result = fit_text("Hello", 200, 100, TEST_FONT)
        assert len(result.lines) == 1
        assert result.lines[0] == "Hello"

    def test_fit_result_has_positive_dimensions_for_nonempty_text(self):
        result = fit_text("Test", 200, 100, TEST_FONT)
        assert result.total_width > 0
        assert result.total_height > 0


class TestRenderer:
    def test_render_translations_returns_same_size_image(self):
        from manga_scanner.typesetting.renderer import render_translations
        from manga_scanner.config import TypesettingConfig

        config = TypesettingConfig(font_path=TEST_FONT)
        base = Image.new("RGB", (400, 600), color=(255, 255, 255))
        boxes = [make_box(10, 10, 200, 80)]
        result = render_translations(base, ["Hello world"], boxes, config)
        assert result.size == (400, 600)

    def test_render_does_not_mutate_base_image(self):
        from manga_scanner.typesetting.renderer import render_translations
        from manga_scanner.config import TypesettingConfig

        config = TypesettingConfig(font_path=TEST_FONT)
        base = Image.new("RGB", (400, 600), color=(200, 200, 200))
        original_pixel = base.getpixel((0, 0))
        boxes = [make_box(10, 10, 300, 100)]
        render_translations(base, ["Some text here"], boxes, config)
        assert base.getpixel((0, 0)) == original_pixel

    def test_render_raises_on_length_mismatch(self):
        from manga_scanner.typesetting.renderer import render_translations
        from manga_scanner.config import TypesettingConfig

        config = TypesettingConfig(font_path=TEST_FONT)
        base = Image.new("RGB", (400, 600))
        with pytest.raises(AssertionError):
            render_translations(base, ["one", "two"], [make_box(0, 0, 100, 50)], config)

    def test_render_skips_empty_string(self):
        from manga_scanner.typesetting.renderer import render_translations
        from manga_scanner.config import TypesettingConfig

        config = TypesettingConfig(font_path=TEST_FONT)
        base = Image.new("RGB", (400, 600), color=(255, 255, 255))
        boxes = [make_box(0, 0, 200, 100)]
        result = render_translations(base, [""], boxes, config)
        assert result.size == (400, 600)


class TestReadingOrderSorter:
    def test_two_bubbles_same_row_right_to_left(self):
        left = make_box(50, 100, 150, 150)
        right = make_box(300, 105, 400, 155)
        result = sort_reading_order([left, right], row_threshold=50)
        assert result[0].x_center > result[1].x_center

    def test_two_rows_top_to_bottom(self):
        top = make_box(100, 50, 200, 100)
        bottom = make_box(100, 400, 200, 450)
        result = sort_reading_order([bottom, top], row_threshold=50)
        assert result[0].y_center < result[1].y_center

    def test_empty_input(self):
        assert sort_reading_order([]) == []

    def test_single_box_returned_unchanged(self):
        box = make_box(10, 10, 100, 100)
        assert sort_reading_order([box]) == [box]

    def test_known_three_box_layout(self):
        top_right = make_box(300, 50, 400, 100)
        top_left = make_box(50, 55, 150, 105)
        bottom = make_box(200, 400, 300, 450)
        result = sort_reading_order([top_left, bottom, top_right], row_threshold=80)
        assert result[0] == top_right
        assert result[1] == top_left
        assert result[2] == bottom


class TestCharacterProfiles:
    def test_load_returns_empty_list_when_missing(self, tmp_path):
        from manga_scanner.translation.models import load_character_profiles
        result = load_character_profiles(tmp_path / "none.json")
        assert result == []

    def test_load_parses_profiles(self, tmp_path):
        import json
        from manga_scanner.translation.models import load_character_profiles, SpeechRegister
        data = {"characters": [
            {"name": "Alice", "jp_name": "アリス", "speech_register": "formal",
             "pronouns": "she/her", "speech_notes": "Polite.", "relationships": {}}
        ]}
        f = tmp_path / "chars.json"
        f.write_text(json.dumps(data))
        profiles = load_character_profiles(f)
        assert len(profiles) == 1
        assert profiles[0].name == "Alice"
        assert profiles[0].speech_register == SpeechRegister.FORMAL

    def test_invalid_register_raises(self, tmp_path):
        import json
        from pydantic import ValidationError
        from manga_scanner.translation.models import load_character_profiles
        data = {"characters": [{"name": "X", "speech_register": "rude"}]}
        f = tmp_path / "chars.json"
        f.write_text(json.dumps(data))
        with pytest.raises(ValidationError):
            load_character_profiles(f)


class TestPromptBuilder:
    def test_build_prompt_returns_two_strings(self):
        from manga_scanner.translation.prompt_builder import build_prompt
        system, user = build_prompt(["行くぞ！"], [], page_number=1)
        assert isinstance(system, str) and len(system) > 0
        assert isinstance(user, str) and len(user) > 0

    def test_user_prompt_contains_numbered_text(self):
        from manga_scanner.translation.prompt_builder import build_prompt
        _, user = build_prompt(["行くぞ！", "待って"], [], page_number=1)
        assert "1." in user
        assert "2." in user
        assert "行くぞ！" in user

    def test_user_prompt_contains_count_constraint(self):
        from manga_scanner.translation.prompt_builder import build_prompt
        _, user = build_prompt(["a", "b", "c"], [], page_number=0)
        assert "exactly 3" in user

    def test_character_block_included_when_profiles_provided(self):
        from manga_scanner.translation.prompt_builder import build_prompt
        from manga_scanner.translation.models import CharacterProfile, SpeechRegister
        profiles = [CharacterProfile(name="Alice", speech_register=SpeechRegister.FORMAL)]
        _, user = build_prompt(["test"], profiles, page_number=1)
        assert "Alice" in user

    def test_empty_texts_returns_valid_strings(self):
        from manga_scanner.translation.prompt_builder import build_prompt
        system, user = build_prompt([], [], page_number=0)
        assert isinstance(system, str)
        assert isinstance(user, str)


class TestTranslatorParsing:
    def _make_translator(self):
        from manga_scanner.config import TranslationConfig
        from manga_scanner.translation.translator import Translator
        return Translator(TranslationConfig())

    def test_parse_clean_json_array(self):
        t = self._make_translator()
        result = t._parse_response('["Hello", "Wait"]', 2)
        assert result == ["Hello", "Wait"]

    def test_parse_array_embedded_in_prose(self):
        t = self._make_translator()
        result = t._parse_response('Here you go: ["Hello"]\nDone.', 1)
        assert result == ["Hello"]

    def test_wrong_count_returns_none(self):
        t = self._make_translator()
        result = t._parse_response('["Hello", "Wait"]', 3)
        assert result is None

    def test_malformed_json_returns_none(self):
        t = self._make_translator()
        result = t._parse_response("not json at all", 1)
        assert result is None

    def test_fallback_returns_jp_texts(self):
        t = self._make_translator()
        jp = ["行くぞ！", "待って"]
        result = t._fallback(jp, "")
        assert result.translations == jp

    def test_connect_error_triggers_fallback(self, mocker):
        import httpx
        from manga_scanner.config import TranslationConfig
        from manga_scanner.translation.translator import Translator
        t = Translator(TranslationConfig())
        mocker.patch.object(t, "_call_ollama", side_effect=httpx.ConnectError("down"))
        result = t.translate_page(["テスト"], [], page_number=0)
        assert result.translations == ["テスト"]
