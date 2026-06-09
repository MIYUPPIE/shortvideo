"""Shared contracts for the SHORTVIDEO pipeline.

Every service imports its boundary types from here and nowhere else. The three
stages communicate through plain dicts that match these dataclasses:

    transcribe  ->  Transcript   (segments + word timings)
    selector    ->  list[Clip]   (chosen reel windows, picked by the LLM)
    cutter      ->  mp4 files     (rendered from Clip + source video)

Keep this module dependency-free (stdlib only) so it stays cheap to import in
gate tests and in any service.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


# --------------------------------------------------------------------------- #
# Transcription
# --------------------------------------------------------------------------- #
@dataclass
class Word:
    word: str
    start: float
    end: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Word":
        return Word(word=d["word"], start=float(d["start"]), end=float(d["end"]))


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "words": [w.to_dict() for w in self.words],
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Segment":
        return Segment(
            start=float(d["start"]),
            end=float(d["end"]),
            text=d["text"],
            words=[Word.from_dict(w) for w in d.get("words", [])],
        )


@dataclass
class Transcript:
    video_path: str
    duration: float
    language: str
    segments: list[Segment] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments).strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_path": self.video_path,
            "duration": self.duration,
            "language": self.language,
            "segments": [s.to_dict() for s in self.segments],
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Transcript":
        return Transcript(
            video_path=d["video_path"],
            duration=float(d["duration"]),
            language=d.get("language", "en"),
            segments=[Segment.from_dict(s) for s in d.get("segments", [])],
        )


# --------------------------------------------------------------------------- #
# Clip selection
# --------------------------------------------------------------------------- #
@dataclass
class Clip:
    index: int
    start: float
    end: float
    title: str
    reason: str = ""
    hook_score: float = 0.0

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["duration"] = self.duration
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Clip":
        return Clip(
            index=int(d.get("index", 0)),
            start=float(d["start"]),
            end=float(d["end"]),
            title=d.get("title", ""),
            reason=d.get("reason", ""),
            hook_score=float(d.get("hook_score", 0.0)),
        )


# JSON Schema handed to the LLM via `claude --json-schema`. The model must
# return exactly this shape; the selector still re-validates everything.
CLIP_SELECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "clips": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "start": {"type": "number", "description": "clip start in seconds"},
                    "end": {"type": "number", "description": "clip end in seconds"},
                    "title": {"type": "string", "description": "punchy reel title"},
                    "reason": {"type": "string", "description": "why this is a strong short"},
                    "hook_score": {
                        "type": "number",
                        "description": "0-100, how strong the hook is",
                    },
                },
                "required": ["start", "end", "title", "hook_score"],
            },
        }
    },
    "required": ["clips"],
}
