import unittest
from unittest.mock import patch

from core.deepseek_client import (
    DEEPSEEK_ANTHROPIC_BASE_URL,
    DEEPSEEK_DEFAULT_MODEL,
    DEEPSEEK_V4_FLASH_TOKENIZER,
    DEEPSEEK_V4_FLASH_TOKENIZER_REVISION,
    DEEPSEEK_V4_PRO_TOKENIZER,
    DEEPSEEK_V4_PRO_TOKENIZER_REVISION,
    deepseek_request_options,
    extract_text,
    load_deepseek_config,
    load_deepseek_tokenizer,
)


class DeepSeekConfigTests(unittest.TestCase):
    def test_deepseek_variables_are_primary(self):
        cfg = load_deepseek_config({
            "DEEPSEEK_API_KEY": "test-deepseek-key",
            "DEEPSEEK_BASE_URL": DEEPSEEK_ANTHROPIC_BASE_URL,
            "DEEPSEEK_MODEL": DEEPSEEK_DEFAULT_MODEL,
            "ANTHROPIC_API_KEY": "legacy-key",
        })
        self.assertEqual("deepseek", cfg["provider"])
        self.assertEqual("test-deepseek-key", cfg["api_key"])
        self.assertEqual(DEEPSEEK_ANTHROPIC_BASE_URL, cfg["base_url"])
        self.assertEqual(DEEPSEEK_DEFAULT_MODEL, cfg["model"])

    def test_legacy_model_name_is_migrated(self):
        cfg = load_deepseek_config({
            "DEEPSEEK_API_KEY": "test-key",
            "DEEPSEEK_MODEL": "deepseek-chat",
        })
        self.assertEqual(DEEPSEEK_DEFAULT_MODEL, cfg["model"])

    def test_missing_key_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "DEEPSEEK_API_KEY"):
            load_deepseek_config({})

    def test_thinking_is_disabled_for_structured_calls(self):
        self.assertEqual(
            {"extra_body": {"thinking": {"type": "disabled"}}},
            deepseek_request_options(),
        )

    def test_tokenizer_repository_matches_the_runtime_model(self):
        sentinel = object()
        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=sentinel,
        ) as loader:
            self.assertIs(sentinel, load_deepseek_tokenizer("deepseek-v4-flash"))
            loader.assert_called_once_with(
                DEEPSEEK_V4_FLASH_TOKENIZER,
                revision=DEEPSEEK_V4_FLASH_TOKENIZER_REVISION,
            )

        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=sentinel,
        ) as loader:
            self.assertIs(sentinel, load_deepseek_tokenizer("deepseek-v4-pro"))
            loader.assert_called_once_with(
                DEEPSEEK_V4_PRO_TOKENIZER,
                revision=DEEPSEEK_V4_PRO_TOKENIZER_REVISION,
            )

    def test_text_extraction_ignores_thinking_blocks(self):
        response = type("Response", (), {
            "content": [
                {"type": "thinking", "thinking": "hidden"},
                {"type": "text", "text": '{"action":"FINAL"}'},
            ],
        })()
        self.assertEqual('{"action":"FINAL"}', extract_text(response))


if __name__ == "__main__":
    unittest.main()
