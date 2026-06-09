"""Periodic eval for the selector (calls the real local Claude).

Quality check, not a gate test: it spends an LLM call, so it runs before ship
and nightly, not on every commit. It feeds a transcript with one obviously
viral moment planted at a known timestamp and checks the model finds it.

Pass threshold: >= 0.66 of the planted "gold" moments are covered by a chosen
clip, AND every returned clip is in-bounds and within the length limits.

Run:
    conda run -n tf_gpu python -m services.selector.evals.eval_selector
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from contracts.schemas import Segment, Transcript  # noqa: E402
from services.selector.select_clips import select_clips  # noqa: E402

PASS_THRESHOLD = 0.66
MIN_DUR, MAX_DUR = 15.0, 60.0

# Synthetic transcript: filler around three strong, self-contained moments.
# Each "gold" moment is a (start, end) window we expect a clip to overlap.
_SEGMENTS = [
    Segment(0.0, 8.0, "Welcome back to the channel, thanks for being here, let's get into it."),
    Segment(8.0, 16.0, "Before we start, a quick reminder to hydrate and stretch, okay."),
    # GOLD 1: a strong, quotable tip
    Segment(16.0, 24.0, "Here is the one habit that doubled my income: I stopped checking email before noon."),
    Segment(24.0, 33.0, "It sounds small but it protected my deep work hours completely, every single day."),
    Segment(33.0, 42.0, "So anyway, moving on, some housekeeping notes about next week's schedule."),
    Segment(42.0, 50.0, "We had a few comments about the lighting, we are working on it, thank you."),
    # GOLD 2: an emotional story beat
    Segment(50.0, 59.0, "The day I almost quit, a stranger emailed me saying my video saved his business."),
    Segment(59.0, 68.0, "I sat in my car and cried, that one message changed everything for me."),
    Segment(68.0, 76.0, "Right, let's talk about the gear I used, nothing too exciting here honestly."),
    # GOLD 3: a surprising, contrarian claim
    Segment(76.0, 85.0, "Posting more often actually hurt my growth, and here is the data that proves it."),
    Segment(85.0, 94.0, "When I cut my uploads in half, watch time per video tripled within a month."),
    Segment(94.0, 102.0, "Okay that's it for today, see you next time, take care everyone."),
]
_GOLD = [(16.0, 33.0), (50.0, 68.0), (76.0, 94.0)]


def _overlap(a, b) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def run_eval() -> bool:
    transcript = Transcript("synthetic.mp4", 102.0, "en", _SEGMENTS)
    clips = select_clips(transcript, n_clips=3, min_dur=MIN_DUR, max_dur=MAX_DUR)

    print(f"model returned {len(clips)} clips:")
    for c in clips:
        print(f"  {c.start:.1f}-{c.end:.1f}s ({c.duration:.0f}s) hook={c.hook_score:.0f}  {c.title}")

    # Validity: in-bounds and within length limits.
    valid = all(
        0.0 <= c.start < c.end <= 102.0 and MIN_DUR - 0.5 <= c.duration <= MAX_DUR + 0.5
        for c in clips
    )

    # Coverage: how many gold moments overlap >= 3s with some chosen clip.
    covered = sum(
        1 for g in _GOLD if any(_overlap((c.start, c.end), g) >= 3.0 for c in clips)
    )
    coverage = covered / len(_GOLD)

    print(f"\nvalid={valid}  coverage={coverage:.2f} ({covered}/{len(_GOLD)})  "
          f"threshold={PASS_THRESHOLD}")
    passed = valid and coverage >= PASS_THRESHOLD
    print("EVAL:", "PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    raise SystemExit(0 if run_eval() else 1)
