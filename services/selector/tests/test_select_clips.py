"""Gate tests for clip validation + the select_clips wiring (LLM mocked)."""
import pytest

from contracts.schemas import Segment, Transcript
from services.selector import select_clips as sc


def _t(duration=120.0):
    return Transcript("v.mp4", duration, "en", [Segment(0, 10, "hi")])


def test_validate_clamps_to_bounds():
    out = sc.validate_clips([{"start": -5, "end": 200, "hook_score": 90}], duration=100, max_dur=300)
    assert len(out) == 1
    assert out[0].start == 0.0 and out[0].end == 100.0


def test_validate_drops_too_short():
    out = sc.validate_clips([{"start": 0, "end": 5, "hook_score": 50}], duration=100, min_dur=15)
    assert out == []


def test_validate_trims_too_long():
    out = sc.validate_clips([{"start": 0, "end": 90, "hook_score": 50}], duration=100, max_dur=60)
    assert out[0].duration == 60.0


def test_validate_dedupes_overlap_keeping_higher_score():
    raw = [
        {"start": 0, "end": 30, "hook_score": 40, "title": "weak"},
        {"start": 10, "end": 40, "hook_score": 95, "title": "strong"},
        {"start": 50, "end": 80, "hook_score": 60, "title": "ok"},
    ]
    out = sc.validate_clips(raw, duration=120)
    assert [c.title for c in out] == ["strong", "ok"]  # weak overlaps strong, dropped
    assert [c.index for c in out] == [0, 1]  # re-indexed
    assert out[0].start < out[1].start  # sorted by start


def test_validate_skips_malformed():
    out = sc.validate_clips([{"start": "x"}, {"end": 10}, {}], duration=100)
    assert out == []


def test_select_clips_uses_validator(monkeypatch):
    fake = {"clips": [
        {"start": 0, "end": 30, "title": "a", "hook_score": 80},
        {"start": 5, "end": 25, "title": "dup", "hook_score": 70},  # overlaps a
        {"start": 40, "end": 70, "title": "b", "hook_score": 90},
    ]}
    monkeypatch.setattr(sc, "call_claude", lambda *a, **k: fake)
    clips = sc.select_clips(_t(), n_clips=5)
    titles = [c.title for c in clips]
    assert "dup" not in titles
    assert set(titles) == {"a", "b"}


def test_select_clips_respects_n(monkeypatch):
    fake = {"clips": [
        {"start": i * 30, "end": i * 30 + 20, "title": f"c{i}", "hook_score": 50 + i}
        for i in range(6)
    ]}
    monkeypatch.setattr(sc, "call_claude", lambda *a, **k: fake)
    clips = sc.select_clips(_t(300), n_clips=3)
    assert len(clips) == 3
