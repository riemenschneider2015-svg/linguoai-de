from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("ttkbootstrap"), "ttkbootstrap not installed")
class GuiHelperTests(unittest.TestCase):
    def test_language_display_and_custom_code_are_parsed(self) -> None:
        from linguoai.gui import LinguoAIApp

        self.assertEqual(LinguoAIApp._code_from_display("Deutsch [de]"), "de")
        self.assertEqual(LinguoAIApp._code_from_display("zh-cn"), "zh-CN")
        self.assertEqual(LinguoAIApp._code_from_display("ace"), "ace")

    def test_file_output_suggestion_contains_current_target(self) -> None:
        from linguoai.gui import LinguoAIApp

        source = Path("C:/Videos/demo.mp4")
        suggestion = LinguoAIApp._suggest_output_path(source, "fr", "file")
        self.assertEqual(Path(suggestion), source.with_name("demo.translated-fr.mp4"))

    def test_folder_output_suggestion_contains_current_target(self) -> None:
        from linguoai.gui import LinguoAIApp

        source = Path("C:/Videos/Eingang")
        suggestion = LinguoAIApp._suggest_output_path(source, "de", "folder")
        self.assertEqual(Path(suggestion), source.with_name("Eingang-translated-de"))

    def test_chatterbox_language_check_handles_regional_codes(self) -> None:
        from linguoai.gui import LinguoAIApp

        self.assertTrue(LinguoAIApp._chatterbox_supports_language("de-DE"))
        self.assertTrue(LinguoAIApp._chatterbox_supports_language("zh-CN"))
        self.assertFalse(LinguoAIApp._chatterbox_supports_language("vi"))

    def test_parakeet_model_helpers_available(self) -> None:
        from linguoai.model_downloader import default_parakeet_directory, is_parakeet_model_ready

        self.assertIsNotNone(default_parakeet_directory())
        self.assertFalse(is_parakeet_model_ready(Path("/nonexistent/path/for/test")))

    def test_check_environment_with_parakeet_does_not_raise_unbound_local(self) -> None:
        import types
        from linguoai.gui import ASR_ENGINE_DISPLAY, LinguoAIApp

        status_value = []
        parakeet_key = [k for k, v in ASR_ENGINE_DISPLAY.items() if v == "parakeet"][0]
        fake_app = types.SimpleNamespace(
            environment_label=types.SimpleNamespace(configure=lambda **kw: None),
            environment_status=types.SimpleNamespace(set=lambda val: status_value.append(val)),
            asr_engine=types.SimpleNamespace(get=lambda: parakeet_key),
            parakeet_model_directory=types.SimpleNamespace(get=lambda: ""),
            _missing_modules=lambda mods: [],
            service=types.SimpleNamespace(get=lambda: "google"),
            optimize=types.SimpleNamespace(get=lambda: False),
            _tts_engine_code=lambda: "edge",
            compute_type=types.SimpleNamespace(get=lambda: "Automatisch"),
            device=types.SimpleNamespace(get=lambda: "Automatisch (empfohlen)"),
        )
        LinguoAIApp._check_environment(fake_app)
        self.assertTrue(any("Parakeet-Modell herunterladen" in s for s in status_value))


if __name__ == "__main__":
    unittest.main()
