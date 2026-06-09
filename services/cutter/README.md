# cutter service

`Clip`s + source video in, rendered reels out.

- `build_cut_cmd` — ffmpeg command for one clip (pure, tested). Re-encodes for
  frame-accurate trims (`-ss start -i src -t dur`).
- `VERTICAL_FILTER` — center-crop to 9:16 then scale to 1080x1920.
- `srt_for_clip` — builds an SRT from the words inside the clip window, times
  rebased to the clip start, grouped into short readable cues.
- `cut_clip` / `cut_all` — run ffmpeg, write `clip_NN_<slug>.mp4` (+ `.srt`).

```bash
python -m services.cutter.cut in.mp4 clips.json -o reels/ \
    --vertical --captions --transcript transcript.json
```

Captions require `--transcript`. Tests: `tests/test_cut.py` (command + SRT
construction) plus `tests/test_cut_integration.py` at repo root (real ffmpeg,
verifies 1080x1920 output).
