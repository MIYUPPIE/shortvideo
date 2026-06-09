"""Cutter service.

Renders each chosen Clip from the source video with ffmpeg. Supports:
  - accurate trim (re-encode, so cuts land exactly on the requested seconds)
  - vertical 9:16 reframe (center-crop then scale to 1080x1920) for Reels
  - optional burned-in captions from the transcript words inside the clip

Command construction is pure (build_cut_cmd / srt_for_clip) so gate tests cover
the ffmpeg arguments without spawning ffmpeg.

Contract in:  contracts.schemas.Clip (+ optional Transcript for captions)
Contract out: mp4 files on disk

CLI:
    python -m services.cutter.cut INPUT.mp4 clips.json -o out/ \
        --vertical --captions --transcript transcript.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from contracts.schemas import Clip, Transcript  # noqa: E402

VERTICAL_FILTER = "crop='min(iw,ih*9/16)':ih,scale=1080:1920,setsar=1"
_CAPTION_STYLE = (
    "FontName=Arial,FontSize=14,Bold=1,PrimaryColour=&H00FFFFFF,"
    "OutlineColour=&H00000000,Outline=2,Shadow=1,Alignment=2,MarginV=60"
)


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def _srt_ts(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def slugify(text: str, fallback: str) -> str:
    s = re.sub(r"[^\w\s-]", "", text).strip().lower()
    s = re.sub(r"[\s_-]+", "-", s)
    return s[:48] or fallback


def srt_for_clip(transcript: Transcript, clip: Clip, max_chars: int = 42) -> str:
    """Build an SRT for the clip window, times rebased to the clip start.

    Groups words into short caption cues so on-screen text stays readable on a
    phone. Words are taken from segments that overlap [clip.start, clip.end].
    """
    words = []
    for seg in transcript.segments:
        if seg.end <= clip.start or seg.start >= clip.end:
            continue
        source = seg.words or [type("W", (), {"word": seg.text, "start": seg.start, "end": seg.end})()]
        for w in source:
            if w.end <= clip.start or w.start >= clip.end:
                continue
            words.append(w)

    cues: list[tuple[float, float, str]] = []
    buf: list[str] = []
    buf_start: float | None = None
    buf_end = 0.0
    for w in words:
        token = w.word.strip()
        if not token:
            continue
        if buf_start is None:
            buf_start = max(w.start, clip.start)
        candidate = (" ".join(buf + [token])).strip()
        if buf and len(candidate) > max_chars:
            cues.append((buf_start - clip.start, buf_end - clip.start, " ".join(buf).strip()))
            buf, buf_start = [token], max(w.start, clip.start)
        else:
            buf.append(token)
        buf_end = min(w.end, clip.end)
    if buf and buf_start is not None:
        cues.append((buf_start - clip.start, buf_end - clip.start, " ".join(buf).strip()))

    out = []
    for i, (start, end, text) in enumerate(cues, 1):
        if end <= start:
            end = start + 0.4
        out.append(f"{i}\n{_srt_ts(start)} --> {_srt_ts(end)}\n{text}\n")
    return "\n".join(out)


def build_cut_cmd(
    src: str,
    clip: Clip,
    out_path: str,
    vertical: bool = True,
    srt_path: str | None = None,
) -> list[str]:
    """ffmpeg command to render one clip. Re-encodes for frame-accurate cuts."""
    filters = []
    if vertical:
        filters.append(VERTICAL_FILTER)
    if srt_path:
        escaped = srt_path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
        filters.append(f"subtitles='{escaped}':force_style='{_CAPTION_STYLE}'")

    cmd = ["ffmpeg", "-y", "-ss", f"{clip.start:.3f}", "-i", src, "-t", f"{clip.duration:.3f}"]
    if filters:
        cmd += ["-vf", ",".join(filters)]
    cmd += [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        out_path,
    ]
    return cmd


# --------------------------------------------------------------------------- #
# Runtime
# --------------------------------------------------------------------------- #
def cut_clip(
    src: str,
    clip: Clip,
    out_dir: str,
    vertical: bool = True,
    transcript: Transcript | None = None,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    base = f"clip_{clip.index:02d}_{slugify(clip.title, f'clip{clip.index}')}"
    out_path = os.path.join(out_dir, base + ".mp4")

    srt_path = None
    if transcript is not None:
        srt = srt_for_clip(transcript, clip)
        if srt.strip():
            srt_path = os.path.join(out_dir, base + ".srt")
            with open(srt_path, "w") as f:
                f.write(srt)

    cmd = build_cut_cmd(src, clip, out_path, vertical=vertical, srt_path=srt_path)
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path


def cut_all(
    src: str,
    clips: list[Clip],
    out_dir: str,
    vertical: bool = True,
    transcript: Transcript | None = None,
) -> list[str]:
    return [cut_clip(src, c, out_dir, vertical=vertical, transcript=transcript) for c in clips]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Cut chosen clips into vertical reels.")
    p.add_argument("video", help="source video")
    p.add_argument("clips", help="clips JSON from the selector service")
    p.add_argument("-o", "--out-dir", default="reels")
    p.add_argument("--vertical", action="store_true", help="reframe to 9:16 1080x1920")
    p.add_argument("--captions", action="store_true", help="burn in captions")
    p.add_argument("--transcript", default=None, help="transcript JSON (required for --captions)")
    args = p.parse_args(argv)

    with open(args.clips) as f:
        clips = [Clip.from_dict(c) for c in json.load(f)]

    transcript = None
    if args.captions:
        if not args.transcript:
            p.error("--captions requires --transcript")
        with open(args.transcript) as f:
            transcript = Transcript.from_dict(json.load(f))

    paths = cut_all(args.video, clips, args.out_dir, vertical=args.vertical, transcript=transcript)
    print(f"[cutter] rendered {len(paths)} reels -> {args.out_dir}")
    for pth in paths:
        print(f"  {pth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
