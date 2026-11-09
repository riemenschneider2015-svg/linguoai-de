from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from linguoai.model_downloader import (
    default_parakeet_directory,
    download_parakeet_model,
    is_parakeet_model_ready,
    sha256_file,
)


class ModelDownloaderTests(unittest.TestCase):
    def test_default_directory_path(self) -> None:
        path = default_parakeet_directory()
        self.assertTrue(str(path).endswith("parakeet-v3"))
        self.assertEqual(path.parent.name, "models")

    def test_is_model_ready_requires_all_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            self.assertFalse(is_parakeet_model_ready(directory))
            self.assertFalse(is_parakeet_model_ready(None))

            (directory / "encoder.int8.onnx").write_bytes(b"enc")
            (directory / "decoder.int8.onnx").write_bytes(b"dec")
            (directory / "joiner.int8.onnx").write_bytes(b"join")
            self.assertFalse(is_parakeet_model_ready(directory))

            (directory / "tokens.txt").write_text("token", encoding="utf-8")
            self.assertTrue(is_parakeet_model_ready(directory))

    def test_sha256_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test.txt"
            file_path.write_bytes(b"hello world")
            # sha256 of "hello world" is b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9
            self.assertEqual(
                sha256_file(file_path),
                "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9",
            )

    def test_download_cancellation(self) -> None:
        cancel_event = threading.Event()
        cancel_event.set()
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch("urllib.request.urlopen") as mock_urlopen:
                mock_response = mock.MagicMock()
                mock_response.__enter__.return_value = mock_response
                mock_response.read.return_value = b'{"sha": "0123456789abcdef0123456789abcdef01234567", "siblings": []}'
                mock_urlopen.return_value = mock_response

                with self.assertRaisesRegex(RuntimeError, "abgebrochen"):
                    download_parakeet_model(Path(temp_dir), cancel_event=cancel_event)


if __name__ == "__main__":
    unittest.main()
