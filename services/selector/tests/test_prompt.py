"""Gate tests for prompt construction."""
from contracts.schemas import Segment, Transcript
from services.selector.prompt import build_prompt, format_transcript


def _t():
    return Transcript("v.mp4", 90.0, "en", [
        Segment(0.0, 3.5, "this changed my life"),
        Segment(3.5, 6.0, ""),  # blank should be skipped
        Segment(6.0, 9.0, "here is why"),
    ])


def test_format_skips_blank_and_keeps_timestamps():
    out = format_transcript(_t())
    assert "[0.00-3.50] this changed my life" in out
    assert "[6.00-9.00] here is why" in out
    assert out.count("\n") == 1  # only two non-blank lines


def test_build_prompt_contains_constraints():
    p = build_prompt(_t(), n_clips=4, min_dur=12, max_dur=45)
    assert "4 BEST" in p
    assert "12" in p and "45" in p
    assert "this changed my life" in p
    assert "hook_score" in p
