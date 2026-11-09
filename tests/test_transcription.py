import struct
import tempfile
import threading
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from linguoai.config import AppConfig
from linguoai.domain import AudioChunk
from linguoai.errors import CancelledError, TranscriptionError
from linguoai.transcription import FasterWhisperTranscriber


class TranscriptionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "chunk.wav"

    def write_wav(self, *, channels=1, sample_width=2, sample_rate=16_000, samples=None):
        if samples is None:
            samples = [-32768, -16384, 0, 16384, 32767] * 3200
        with wave.open(str(self.path), "wb") as writer:
            writer.setnchannels(channels)
            writer.setsampwidth(sample_width)
            writer.setframerate(sample_rate)
            writer.writeframes(struct.pack(f"<{len(samples)}h", *samples))

    def chunk(self, *, index=0, start=0):
        return AudioChunk(index, self.path, start, start + 1, start, start + 1)

    @staticmethod
    def model():
        # Reproduce the upstream failure for filenames, while accepting its
        # documented waveform input. No model download or external audio needed.
        def transcribe(audio, **_kwargs):
            if not isinstance(audio, np.ndarray):
                raise TypeError("open() got an unexpected keyword argument 'metadata_errors'")
            return iter([SimpleNamespace(start=0.1, end=0.8, text=" Hallo ")]), SimpleNamespace(
                language="de", language_probability=0.98
            )

        return SimpleNamespace(transcribe=mock.Mock(side_effect=transcribe))

    def test_pcm_waveform_bypasses_incompatible_decoder_and_preserves_samples(self):
        self.write_wav()
        model = self.model()
        adapter = FasterWhisperTranscriber(AppConfig(), model=model)
        result = adapter.transcribe(
            [self.chunk()], source_language=None, cancel_event=threading.Event()
        )
        audio = model.transcribe.call_args.args[0]
        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(audio.shape, (16_000,))
        np.testing.assert_array_equal(
            audio[:5], np.array([-1, -0.5, 0, 0.5, 32767 / 32768], dtype=np.float32)
        )
        self.assertEqual(result.language, "de")
        self.assertEqual(result.segments[0].text, "Hallo")
        self.assertEqual(result.segments[0].start, 0.1)

    def test_parallel_chunks_keep_offsets_without_path_decoding(self):
        self.write_wav()
        adapter = FasterWhisperTranscriber(AppConfig(transcription_workers=2), model=self.model())
        result = adapter.transcribe(
            [self.chunk(), self.chunk(index=1, start=2)],
            source_language="de-DE",
            cancel_event=threading.Event(),
        )
        self.assertEqual([segment.start for segment in result.segments], [0.1, 2.1])
        self.assertEqual([segment.end for segment in result.segments], [0.8, 2.8])

    def test_invalid_audio_never_reaches_model(self):
        for options in ({"channels": 2}, {"sample_width": 1}, {"sample_rate": 48_000}):
            with self.subTest(options=options):
                self.write_wav(**options)
                model = self.model()
                adapter = FasterWhisperTranscriber(AppConfig(), model=model)
                with self.assertRaisesRegex(TranscriptionError, "Mono-WAV"):
                    adapter._transcribe_one(self.chunk(), "de", threading.Event())
                model.transcribe.assert_not_called()

    def test_empty_and_truncated_audio_are_rejected(self):
        self.write_wav(samples=[])
        with self.assertRaisesRegex(TranscriptionError, "leer oder unvollständig"):
            FasterWhisperTranscriber._read_pcm_chunk(self.path)
        self.write_wav()
        self.path.write_bytes(self.path.read_bytes()[:-2])
        with self.assertRaisesRegex(TranscriptionError, "leer oder unvollständig"):
            FasterWhisperTranscriber._read_pcm_chunk(self.path)

    def test_cancellation_stops_before_audio_read(self):
        cancel = threading.Event()
        cancel.set()
        model = self.model()
        adapter = FasterWhisperTranscriber(AppConfig(), model=model)
        with self.assertRaises(CancelledError):
            adapter._transcribe_one(self.chunk(), "de", cancel)
        model.transcribe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
