"""Transcription service.

Extracts audio with ffmpeg, then runs faster-whisper (GPU) to produce a
word-timestamped Transcript. Heavy imports (faster_whisper) live inside the
functions that need them so the pure helpers below import instantly in gate
tests.

Contract out: contracts.schemas.Transcript

CLI:
    python -m services.transcribe.transcribe INPUT.mp4 -o transcript.json \
        --model large-v3 --device cuda --compute-type float16
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

# Make `contracts` importable whether run as a module or a script.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from contracts.schemas import Segment, Transcript, Word  # noqa: E402


# --------------------------------------------------------------------------- #
# Pure helpers (deterministic, no GPU, gate-testable)
# --------------------------------------------------------------------------- #
def build_extract_cmd(video_path: str, wav_path: str, sample_rate: int = 16000) -> list[str]:
    """ffmpeg command that downmixes any input to mono 16kHz PCM wav."""
    return [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",                       # drop video
        "-ac", "1",                  # mono
        "-ar", str(sample_rate),     # 16 kHz is what whisper expects
        "-f", "wav",
        wav_path,
    ]


def probe_duration(video_path: str) -> float:
    """Container duration in seconds via ffprobe. 0.0 if it can't be read."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
        return round(float(out), 3)
    except (subprocess.CalledProcessError, ValueError):
        return 0.0


# --------------------------------------------------------------------------- #
# Runtime (needs ffmpeg + faster-whisper + GPU)
# --------------------------------------------------------------------------- #
def extract_audio(video_path: str, wav_path: str, sample_rate: int = 16000) -> str:
    subprocess.run(
        build_extract_cmd(video_path, wav_path, sample_rate),
        check=True,
        capture_output=True,
    )
    return wav_path


def transcribe_video(
    video_path: str,
    model_size: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "float16",
    language: str | None = None,
    beam_size: int = 5,
) -> Transcript:
    """Transcribe a video to a word-timestamped Transcript."""
    from faster_whisper import WhisperModel  # heavy import, kept local

    if not os.path.isfile(video_path):
        raise FileNotFoundError(video_path)

    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "audio.wav")
        extract_audio(video_path, wav)

        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        seg_iter, info = model.transcribe(
            wav,
            language=language,
            beam_size=beam_size,
            word_timestamps=True,
            vad_filter=True,
        )

        segments: list[Segment] = []
        for s in seg_iter:
            words = [
                Word(word=w.word, start=round(w.start, 3), end=round(w.end, 3))
                for w in (s.words or [])
            ]
            segments.append(
                Segment(
                    start=round(s.start, 3),
                    end=round(s.end, 3),
                    text=s.text.strip(),
                    words=words,
                )
            )

    duration = probe_duration(video_path) or (segments[-1].end if segments else 0.0)
    return Transcript(
        video_path=os.path.abspath(video_path),
        duration=duration,
        language=getattr(info, "language", language or "en"),
        segments=segments,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Transcribe a video with faster-whisper (GPU).")
    p.add_argument("video", help="input video file")
    p.add_argument("-o", "--out", default="transcript.json", help="output JSON path")
    p.add_argument("--model", default="large-v3", help="whisper model size")
    p.add_argument("--device", default="cuda")
    p.add_argument("--compute-type", default="float16")
    p.add_argument("--language", default=None, help="force language code (auto-detect if unset)")
    args = p.parse_args(argv)

    t = transcribe_video(
        args.video,
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
    )
    with open(args.out, "w") as f:
        json.dump(t.to_dict(), f, indent=2, ensure_ascii=False)
    print(f"[transcribe] {len(t.segments)} segments, {t.duration:.1f}s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
