# transcribe service

Video in, word-timestamped `Transcript` out.

- `build_extract_cmd` — ffmpeg cmd to make mono 16 kHz wav (pure, tested).
- `probe_duration` — container length via ffprobe.
- `transcribe_video` — extracts audio, runs faster-whisper (`device=cuda`,
  `compute_type=float16`) with `word_timestamps=True` and VAD filtering.

`faster_whisper` is imported lazily inside `transcribe_video`, so the pure
helpers (and gate tests) import with no GPU and no model load.

```bash
python -m services.transcribe.transcribe in.mp4 -o transcript.json --model large-v3
```

Contract out: [contracts.schemas.Transcript](../../contracts/schemas.py).
Tests: `tests/test_transcribe.py`.
