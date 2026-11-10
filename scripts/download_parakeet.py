"""Download the documented sherpa-onnx NVIDIA v3 INT8 export, pinned and verified.

No arbitrary repository or executable is accepted. Downloads are atomic and each
large model file must match the publisher's LFS SHA-256 before it becomes usable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from linguoai.model_downloader import (
    PARAKEET_V3_FILES,
    PARAKEET_V3_REPOSITORY,
    default_parakeet_directory,
    download_parakeet_model,
    is_parakeet_model_ready,
    sha256_file,
)

REPOSITORY = PARAKEET_V3_REPOSITORY
MODEL_FILES = PARAKEET_V3_FILES


def download(directory: Path, *, samples: bool = False) -> dict:
    files = (*MODEL_FILES, *(("test_wavs/en.wav", "test_wavs/de.wav") if samples else ()))
    target = download_parakeet_model(
        directory,
        repository=REPOSITORY,
        files=files,
        progress_callback=lambda name, frac, done, total: print(
            f"Downloading {name}: {done / (1024 * 1024):.1f} MB ({int(frac * 100)}%)", flush=True
        ),
    )
    import json
    return json.loads((target / "MODEL_MANIFEST.json").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Install verified Parakeet v3 CPU model")
    parser.add_argument("--directory", type=Path, default=default_parakeet_directory())
    parser.add_argument("--with-samples", action="store_true")
    args = parser.parse_args()
    download(args.directory, samples=args.with_samples)


if __name__ == "__main__":
    main()
