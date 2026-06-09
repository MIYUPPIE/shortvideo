"""Prompt construction for clip selection (pure, deterministic)."""
from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from contracts.schemas import Transcript  # noqa: E402


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def format_transcript(transcript: Transcript) -> str:
    """Compact, timestamped transcript the model reads. One line per segment:

        [START-END] text
    where times are seconds with two decimals so the model can pick exact cuts.
    """
    lines = []
    for seg in transcript.segments:
        if not seg.text.strip():
            continue
        lines.append(f"[{seg.start:.2f}-{seg.end:.2f}] {seg.text.strip()}")
    return "\n".join(lines)


SYSTEM = (
    "You are a senior short-form video editor who finds the moments that go "
    "viral as Reels, Shorts, and TikToks. You think in hooks, payoffs, and "
    "self-contained ideas."
)


def build_prompt(
    transcript: Transcript,
    n_clips: int = 5,
    min_dur: float = 15.0,
    max_dur: float = 60.0,
) -> str:
    """Full user prompt. The schema (passed separately) enforces output shape."""
    body = format_transcript(transcript)
    total = transcript.duration
    return f"""{SYSTEM}

Below is a timestamped transcript of a {total:.0f}-second video. Each line is
`[start-end] text` with times in seconds from the start of the video.

TASK: Pick the {n_clips} BEST standalone short clips for vertical social video.

Rules for each clip:
- Length between {min_dur:.0f} and {max_dur:.0f} seconds.
- Must start on a strong hook and end on a clear payoff or punchline. Do not cut
  mid-sentence; align `start` and `end` to the segment boundaries shown.
- Must make sense on its own with zero outside context.
- Prefer emotion, surprise, a strong opinion, a concrete tip, or a story.
- Clips must not overlap each other.
- `start` and `end` are seconds (floats) within [0, {total:.2f}].
- `hook_score` is 0-100 for how strongly the first 3 seconds grab attention.
- `title` is a punchy 3-8 word caption. `reason` is one sentence on why it wins.

Return only the structured JSON object of clips.

TRANSCRIPT:
{body}
"""
