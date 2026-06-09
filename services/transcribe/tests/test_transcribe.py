"""Gate tests for transcribe pure helpers (no GPU, no whisper import)."""
from services.transcribe.transcribe import build_extract_cmd


def test_extract_cmd_is_mono_16k_wav():
    cmd = build_extract_cmd("in.mp4", "out.wav")
    assert cmd[0] == "ffmpeg"
    assert "-vn" in cmd
    assert cmd[cmd.index("-ac") + 1] == "1"
    assert cmd[cmd.index("-ar") + 1] == "16000"
    assert cmd[-1] == "out.wav"


def test_extract_cmd_custom_rate():
    cmd = build_extract_cmd("in.mkv", "o.wav", sample_rate=8000)
    assert cmd[cmd.index("-ar") + 1] == "8000"
