from __future__ import annotations

import io
import json
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
            self.assertEqual(
                sha256_file(file_path),
                "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9",
            )

    def test_download_cancellation(self) -> None:
        cancel_event = threading.Event()
        cancel_event.set()
        metadata = json.dumps({"sha": "a" * 40, "siblings": []}).encode()
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            mock.patch("urllib.request.urlopen", return_value=io.BytesIO(metadata)),
            self.assertRaisesRegex(RuntimeError, "abgebrochen"),
        ):
            download_parakeet_model(Path(temp_dir), cancel_event=cancel_event)

    def test_force_downloads_even_if_existing_file_is_valid(self) -> None:
        name = "tokens.txt"
        content = b"test tokens"
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            target = directory / name
            target.write_bytes(content)
            metadata = json.dumps(
                {
                    "sha": "a" * 40,
                    "siblings": [
                        {
                            "rfilename": name,
                            "size": len(content),
                            "lfs": {"sha256": sha256_file(target)},
                        }
                    ],
                }
            ).encode()
            with mock.patch(
                "urllib.request.urlopen",
                side_effect=[io.BytesIO(metadata), io.BytesIO(content)],
            ) as open_url:
                download_parakeet_model(directory, files=(name,), force=True)
            self.assertEqual(open_url.call_count, 2)
            self.assertEqual(target.read_bytes(), content)


if __name__ == "__main__":
    unittest.main()
