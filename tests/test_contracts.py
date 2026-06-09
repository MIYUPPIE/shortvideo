"""Gate tests for the shared contracts: serialization round-trips."""
from contracts.schemas import Clip, Segment, Transcript, Word, CLIP_SELECTION_SCHEMA


def test_transcript_roundtrip():
    t = Transcript(
        video_path="/x/a.mp4",
        duration=12.5,
        language="en",
        segments=[
            Segment(0.0, 2.0, "hello world", [Word("hello", 0.0, 1.0), Word("world", 1.0, 2.0)]),
            Segment(2.0, 4.0, "next bit", []),
        ],
    )
    again = Transcript.from_dict(t.to_dict())
    assert again.duration == 12.5
    assert again.language == "en"
    assert len(again.segments) == 2
    assert again.segments[0].words[1].word == "world"
    assert again.text == "hello world next bit"


def test_clip_duration_and_roundtrip():
    c = Clip(index=3, start=10.0, end=28.5, title="big idea", reason="r", hook_score=88.0)
    assert c.duration == 18.5
    d = c.to_dict()
    assert d["duration"] == 18.5
    again = Clip.from_dict(d)
    assert again.index == 3 and again.title == "big idea" and again.hook_score == 88.0


def test_schema_shape():
    props = CLIP_SELECTION_SCHEMA["properties"]["clips"]["items"]["properties"]
    assert {"start", "end", "title", "hook_score"} <= set(props)
    assert CLIP_SELECTION_SCHEMA["required"] == ["clips"]
