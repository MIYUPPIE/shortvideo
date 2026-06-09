"""Gate tests for cutter command + caption construction (no ffmpeg spawn)."""
from contracts.schemas import Clip, Segment, Transcript, Word
from services.cutter.cut import (
    build_cut_cmd,
    srt_for_clip,
    slugify,
    _srt_ts,
    VERTICAL_FILTER,
)


def _clip():
    return Clip(index=2, start=10.0, end=28.0, title="My Big Idea!", hook_score=80)


def test_cut_cmd_accurate_trim():
    cmd = build_cut_cmd("src.mp4", _clip(), "out.mp4", vertical=False, srt_path=None)
    assert cmd[cmd.index("-ss") + 1] == "10.000"
    assert cmd[cmd.index("-t") + 1] == "18.000"
    assert "libx264" in cmd and cmd[-1] == "out.mp4"
    assert "-vf" not in cmd  # no filters when plain


def test_cut_cmd_vertical_filter():
    cmd = build_cut_cmd("src.mp4", _clip(), "out.mp4", vertical=True, srt_path=None)
    vf = cmd[cmd.index("-vf") + 1]
    assert vf == VERTICAL_FILTER


def test_cut_cmd_captions_chain():
    cmd = build_cut_cmd("src.mp4", _clip(), "out.mp4", vertical=True, srt_path="a/b.srt")
    vf = cmd[cmd.index("-vf") + 1]
    assert VERTICAL_FILTER in vf
    assert "subtitles=" in vf and "b.srt" in vf


def test_slugify():
    assert slugify("My Big Idea!", "fb") == "my-big-idea"
    assert slugify("***", "fb") == "fb"


def test_srt_ts():
    assert _srt_ts(0) == "00:00:00,000"
    assert _srt_ts(3661.5) == "01:01:01,500"


def test_srt_for_clip_rebases_to_zero_and_chunks():
    t = Transcript("v.mp4", 60, "en", [
        Segment(8, 14, "the secret is consistency over time", [
            Word("the", 8.0, 8.3), Word("secret", 8.3, 8.9), Word("is", 8.9, 9.1),
            Word("consistency", 9.1, 10.2), Word("over", 10.2, 10.6), Word("time", 10.6, 11.0),
        ]),
    ])
    clip = Clip(index=0, start=9.0, end=12.0, title="t", hook_score=50)
    srt = srt_for_clip(t, clip, max_chars=20)
    # words before 9.0s ("the","secret") are excluded; first kept word rebased near 0
    assert "the secret" not in srt
    assert "consistency" in srt
    assert srt.startswith("1\n00:00:00")


def test_srt_empty_when_no_overlap():
    t = Transcript("v.mp4", 60, "en", [Segment(0, 5, "hi", [Word("hi", 0, 5)])])
    clip = Clip(index=0, start=20, end=40, title="t", hook_score=1)
    assert srt_for_clip(t, clip).strip() == ""
