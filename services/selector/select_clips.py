"""Clip selection service.

Latent part: the LLM reads the transcript and proposes clips (llm_client).
Deterministic part: validate_clips() clamps, filters, de-overlaps and sorts the
proposals so the cutter always gets clean, in-bounds windows. The deterministic
part is what gate tests cover; the latent part is covered by evals.

Contract in:  contracts.schemas.Transcript
Contract out: list[contracts.schemas.Clip]

CLI:
    python -m services.selector.select_clips transcript.json -o clips.json -n 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from contracts.schemas import CLIP_SELECTION_SCHEMA, Clip, Transcript  # noqa: E402
from services.selector.llm_client import DEFAULT_MODEL, call_claude  # noqa: E402
from services.selector.prompt import build_prompt  # noqa: E402


def _overlaps(a: Clip, b: Clip) -> bool:
    return a.start < b.end and b.start < a.end


def validate_clips(
    raw: list[dict],
    duration: float,
    min_dur: float = 15.0,
    max_dur: float = 60.0,
) -> list[Clip]:
    """Turn raw LLM proposals into clean, non-overlapping, in-bounds clips.

    Steps, in order:
      1. coerce + clamp start/end into [0, duration]
      2. drop anything shorter than min_dur; trim anything longer than max_dur
      3. sort by hook_score (desc) and greedily drop clips overlapping a kept one
      4. re-sort by start time and re-index from 0
    """
    candidates: list[Clip] = []
    for item in raw:
        try:
            start = max(0.0, float(item["start"]))
            end = min(duration if duration > 0 else float(item["end"]), float(item["end"]))
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start:
            continue
        if end - start > max_dur:
            end = start + max_dur
        if end - start < min_dur:
            continue
        candidates.append(
            Clip(
                index=0,
                start=round(start, 3),
                end=round(end, 3),
                title=str(item.get("title", "")).strip(),
                reason=str(item.get("reason", "")).strip(),
                hook_score=float(item.get("hook_score", 0.0) or 0.0),
            )
        )

    # Greedy de-overlap: keep the strongest hook first.
    candidates.sort(key=lambda c: c.hook_score, reverse=True)
    kept: list[Clip] = []
    for c in candidates:
        if not any(_overlaps(c, k) for k in kept):
            kept.append(c)

    kept.sort(key=lambda c: c.start)
    for i, c in enumerate(kept):
        c.index = i
    return kept


def select_clips(
    transcript: Transcript,
    n_clips: int = 5,
    min_dur: float = 15.0,
    max_dur: float = 60.0,
    model: str = DEFAULT_MODEL,
    claude_bin: str | None = None,
) -> list[Clip]:
    prompt = build_prompt(transcript, n_clips=n_clips, min_dur=min_dur, max_dur=max_dur)
    result = call_claude(
        prompt, schema=CLIP_SELECTION_SCHEMA, model=model, claude_bin=claude_bin
    )
    raw = result.get("clips", []) if isinstance(result, dict) else []
    clips = validate_clips(raw, transcript.duration, min_dur=min_dur, max_dur=max_dur)
    return clips[:n_clips]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Pick the best short clips from a transcript.")
    p.add_argument("transcript", help="transcript JSON from the transcribe service")
    p.add_argument("-o", "--out", default="clips.json")
    p.add_argument("-n", "--n-clips", type=int, default=5)
    p.add_argument("--min-dur", type=float, default=15.0)
    p.add_argument("--max-dur", type=float, default=60.0)
    p.add_argument("--model", default=DEFAULT_MODEL)
    args = p.parse_args(argv)

    with open(args.transcript) as f:
        transcript = Transcript.from_dict(json.load(f))

    clips = select_clips(
        transcript,
        n_clips=args.n_clips,
        min_dur=args.min_dur,
        max_dur=args.max_dur,
        model=args.model,
    )
    with open(args.out, "w") as f:
        json.dump([c.to_dict() for c in clips], f, indent=2, ensure_ascii=False)
    print(f"[selector] chose {len(clips)} clips -> {args.out}")
    for c in clips:
        print(f"  #{c.index} {c.start:.1f}-{c.end:.1f}s ({c.duration:.0f}s) "
              f"hook={c.hook_score:.0f}  {c.title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
