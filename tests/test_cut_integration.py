"""Integration test: generate a real video, cut a real clip with ffmpeg.

Validates the cutter ffmpeg path end-to-end (trim accuracy + vertical reframe)
without needing the GPU or the LLM. Skips automatically if ffmpeg/ffprobe are
missing. Still fast (~a couple seconds).
"""
import os
import shutil
import subprocess
import tempfile

import pytest

from contracts.schemas import Clip
from services.cutter.cut import cut_clip

pytestmark = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
    reason="ffmpeg/ffprobe not installed",
)


def _probe_duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def _make_test_video(path: str, seconds: int = 12):
    subprocess.run(
        ["ffmpeg", "-y",
         "-f", "lavfi", "-i", f"testsrc=size=640x360:rate=24:duration={seconds}",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
         "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", path],
        check=True, capture_output=True,
    )


def test_cut_clip_vertical_real_ffmpeg():
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "src.mp4")
        _make_test_video(src, seconds=12)

        clip = Clip(index=0, start=3.0, end=9.0, title="Test Reel", hook_score=77)
        out = cut_clip(src, clip, os.path.join(tmp, "reels"), vertical=True, transcript=None)

        assert os.path.isfile(out)
        assert out.endswith("clip_00_test-reel.mp4")
        dur = _probe_duration(out)
        assert abs(dur - 6.0) < 0.5  # 3s..9s window, frame-accurate within tolerance

        # confirm 9:16 vertical geometry
        dims = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", out],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        assert dims == "1080x1920"
