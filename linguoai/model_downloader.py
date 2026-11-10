"""Download and verification helper for local sherpa-onnx Parakeet models."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path

PARAKEET_V3_REPOSITORY = "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
PARAKEET_V3_FILES = ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")


def default_parakeet_directory() -> Path:
    """Return the default system directory for storing downloaded Parakeet models."""
    return Path.home() / ".linguoai" / "models" / "parakeet-v3"


def is_parakeet_model_ready(directory: Path | None) -> bool:
    """Check whether a directory contains a complete, usable Parakeet sherpa-onnx export."""
    if directory is None:
        return False
    target = Path(directory).expanduser()
    if not target.is_dir():
        return False

    has_encoder = (target / "encoder.int8.onnx").is_file() or (target / "encoder.onnx").is_file()
    has_decoder = (target / "decoder.int8.onnx").is_file() or (target / "decoder.onnx").is_file()
    has_joiner = (target / "joiner.int8.onnx").is_file() or (target / "joiner.onnx").is_file()
    has_tokens = (target / "tokens.txt").is_file()
    return bool(has_encoder and has_decoder and has_joiner and has_tokens)


def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 digest of a local file in 1 MB blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def download_parakeet_model(
    directory: Path | None = None,
    *,
    repository: str = PARAKEET_V3_REPOSITORY,
    files: tuple[str, ...] = PARAKEET_V3_FILES,
    force: bool = False,
    progress_callback: Callable[[str, float, int, int], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> Path:
    """Download and verify the sherpa-onnx Parakeet INT8 export from Hugging Face."""
    target_dir = (directory or default_parakeet_directory()).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    api_url = f"https://huggingface.co/api/models/{repository}?blobs=true"
    request = urllib.request.Request(api_url, headers={"User-Agent": "LinguoAI-ModelDownloader"})
    with urllib.request.urlopen(request, timeout=90) as response:
        metadata = json.load(response)

    revision = metadata["sha"]
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise RuntimeError("Ungültige Modell-Revision vom Server erhalten.")

    published = {item["rfilename"]: item for item in metadata.get("siblings", [])}

    manifest = {
        "repository": repository,
        "revision": revision,
        "files": {},
        "attribution": (
            "NVIDIA parakeet-tdt-0.6b-v3; sherpa-onnx conversion by csukuangfj; CC-BY-4.0"
        ),
        "model_url": "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3",
    }

    for name in files:
        if cancel_event and cancel_event.is_set():
            raise RuntimeError("Download durch Benutzer abgebrochen.")

        info = published.get(name)
        if not info:
            raise RuntimeError(f"Erforderliche Modelldatei nicht gefunden: {name}")

        expected = info.get("lfs", {}).get("sha256")
        size = info.get("size", info.get("lfs", {}).get("size")) or 0

        target_file = target_dir / name
        target_file.parent.mkdir(parents=True, exist_ok=True)

        if (
            not force
            and target_file.is_file()
            and (not size or target_file.stat().st_size == size)
        ):
            digest = sha256_file(target_file)
            if not expected or digest == expected:
                cached_size = target_file.stat().st_size
                manifest["files"][name] = {"sha256": digest, "size": cached_size}
                if progress_callback:
                    progress_callback(name, 1.0, cached_size, cached_size)
                continue

        partial = target_file.with_name(target_file.name + ".partial")
        for attempt in range(3):
            if cancel_event and cancel_event.is_set():
                partial.unlink(missing_ok=True)
                raise RuntimeError("Download durch Benutzer abgebrochen.")

            try:
                digest = hashlib.sha256()
                consumed = 0
                download_url = f"https://huggingface.co/{repository}/resolve/{revision}/{name}"
                dl_req = urllib.request.Request(
                    download_url, headers={"User-Agent": "LinguoAI-ModelDownloader"}
                )
                with urllib.request.urlopen(dl_req, timeout=90) as resp, partial.open("wb") as out:
                    while block := resp.read(1024 * 1024):
                        if cancel_event and cancel_event.is_set():
                            raise RuntimeError("Download durch Benutzer abgebrochen.")
                        consumed += len(block)
                        if consumed > 900 * 1024**2:
                            raise RuntimeError(f"Dateigröße überschreitet Limit: {name}")
                        digest.update(block)
                        out.write(block)
                        if progress_callback:
                            fraction = (consumed / size) if size else 0.0
                            progress_callback(name, min(fraction, 1.0), consumed, size)

                actual = digest.hexdigest()
                if size and consumed != size:
                    raise RuntimeError(f"Unerwartete Dateigröße für {name}: {consumed} != {size}")
                if expected and actual != expected:
                    raise RuntimeError(f"Prüfsummenfehler für {name}")

                os.replace(partial, target_file)
                manifest["files"][name] = {"sha256": actual, "size": consumed}
                break
            except Exception:
                partial.unlink(missing_ok=True)
                if attempt == 2:
                    raise
                time.sleep(2**attempt)

    (target_dir / "MODEL_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return target_dir


def main() -> None:
    """CLI entrypoint to download the default Parakeet model."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="LinguoAI Parakeet-Modell herunterladen")
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Zielordner (Standard: ~/.linguoai/models/parakeet-v3)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Bereits vorhandenes Modell überschreiben"
    )
    args = parser.parse_args()

    dest = args.output_dir or default_parakeet_directory()
    if is_parakeet_model_ready(dest) and not args.force:
        print(f"Modell ist bereits vollständig vorhanden: {dest}")
        return

    print(f"Lade Parakeet-Modell ({PARAKEET_V3_REPOSITORY}) nach: {dest}")

    def _prog(name: str, frac: float, done: int, total: int) -> None:
        mb_done = done / (1024 * 1024)
        mb_total = total / (1024 * 1024)
        pct = frac * 100
        sys.stdout.write(f"\r  [{name}] {pct:5.1f}% ({mb_done:6.1f} / {mb_total:6.1f} MB)")
        sys.stdout.flush()

    try:
        target = download_parakeet_model(dest, force=args.force, progress_callback=_prog)
        print(f"\n✓ Fertig! Modell bereit unter: {target}")
    except Exception as exc:
        print(f"\n✗ Fehler beim Herunterladen: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

