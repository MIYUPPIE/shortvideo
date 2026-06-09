"""End-to-end orchestration: video in, vertical reels out.

    transcribe (faster-whisper GPU) -> selector (local Claude) -> cutter (ffmpeg)

This is glue only. All logic lives in the services. Run inside tf_gpu:

    conda run -n tf_gpu python pipeline.py INPUT.mp4 -o out/ --n-clips 5 \
        --vertical --captions --model large-v3

Outputs in --out-dir:
    transcript.json   clips.json   reels/clip_00_*.mp4 ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.abspath(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from contracts.schemas import Clip, Transcript  # noqa: E402
from services.transcribe.transcribe import transcribe_video  # noqa: E402
from services.selector.select_clips import select_clips  # noqa: E402
from services.selector.llm_client import DEFAULT_MODEL  # noqa: E402
from services.cutter.cut import cut_all  # noqa: E402


def run(
    video: str,
    out_dir: str,
    n_clips: int = 5,
    min_dur: float = 15.0,
    max_dur: float = 60.0,
    whisper_model: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "float16",
    language: str | None = None,
    llm_model: str = DEFAULT_MODEL,
    vertical: bool = True,
    captions: bool = True,
    reuse_transcript: bool = True,
) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    transcript_path = os.path.join(out_dir, "transcript.json")
    clips_path = os.path.join(out_dir, "clips.json")
    reels_dir = os.path.join(out_dir, "reels")

    # 1. transcribe (cache to disk; whisper is the slow part)
    if reuse_transcript and os.path.isfile(transcript_path):
        print(f"[pipeline] reusing {transcript_path}")
        with open(transcript_path) as f:
            transcript = Transcript.from_dict(json.load(f))
    else:
        print("[pipeline] transcribing...")
        transcript = transcribe_video(
            video, model_size=whisper_model, device=device,
            compute_type=compute_type, language=language,
        )
        with open(transcript_path, "w") as f:
            json.dump(transcript.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"[pipeline] {len(transcript.segments)} segments -> {transcript_path}")

    # 2. select clips with the local LLM
    print("[pipeline] selecting clips...")
    clips = select_clips(
        transcript, n_clips=n_clips, min_dur=min_dur, max_dur=max_dur, model=llm_model,
    )
    with open(clips_path, "w") as f:
        json.dump([c.to_dict() for c in clips], f, indent=2, ensure_ascii=False)
    print(f"[pipeline] {len(clips)} clips -> {clips_path}")
    for c in clips:
        print(f"  #{c.index} {c.start:.1f}-{c.end:.1f}s ({c.duration:.0f}s) "
              f"hook={c.hook_score:.0f}  {c.title}")

    if not clips:
        print("[pipeline] no clips selected, nothing to cut")
        return []

    # 3. cut reels
    print("[pipeline] cutting reels...")
    paths = cut_all(
        video, clips, reels_dir, vertical=vertical,
        transcript=transcript if captions else None,
    )
    print(f"[pipeline] done: {len(paths)} reels in {reels_dir}")
    return paths


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Automated short-video reel cutter.")
    p.add_argument("video", help="input video file")
    p.add_argument("-o", "--out-dir", default="out")
    p.add_argument("-n", "--n-clips", type=int, default=5)
    p.add_argument("--min-dur", type=float, default=15.0)
    p.add_argument("--max-dur", type=float, default=60.0)
    p.add_argument("--model", default="large-v3", help="whisper model size")
    p.add_argument("--device", default="cuda")
    p.add_argument("--compute-type", default="float16")
    p.add_argument("--language", default=None)
    p.add_argument("--llm-model", default=DEFAULT_MODEL)
    p.add_argument("--vertical", action="store_true", help="reframe to 9:16")
    p.add_argument("--captions", action="store_true", help="burn captions")
    p.add_argument("--no-reuse", action="store_true", help="force re-transcribe")
    args = p.parse_args(argv)

    run(
        args.video, args.out_dir,
        n_clips=args.n_clips, min_dur=args.min_dur, max_dur=args.max_dur,
        whisper_model=args.model, device=args.device, compute_type=args.compute_type,
        language=args.language, llm_model=args.llm_model,
        vertical=args.vertical, captions=args.captions,
        reuse_transcript=not args.no_reuse,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
