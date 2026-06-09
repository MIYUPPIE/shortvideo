# SHORTVIDEO — local automated reel cutter

Turn a long talking-head video into short vertical reels, fully local. It
transcribes the video on the GPU, asks a local LLM to pick the best standalone
moments, and cuts each one into a 9:16 captioned clip with ffmpeg.

```
video.mp4 ─▶ transcribe ─▶ selector ─▶ cutter ─▶ reels/*.mp4
            (faster-whisper  (local Claude   (ffmpeg
             GPU, words)      picks clips)    9:16 + captions)
```

No hosted APIs. Transcription runs on your GPU; clip selection shells out to the
local `claude` CLI (the binary shipped with the VSCode extension).

## Requirements

- The `tf_gpu` conda env (Python 3.10). It already has `faster-whisper`,
  `torch` (CUDA), `ctranslate2`, and cuDNN.
- `ffmpeg` + `ffprobe` on PATH.
- The local `claude` CLI. Auto-discovered from PATH, `$CLAUDE_BIN`, or the
  VSCode extension at `~/.vscode/extensions/anthropic.claude-code-*/.../claude`.
- For the one-shot install in this env: `pip install -r requirements.txt`.

## Quick start

```bash
./run.sh INPUT.mp4 -o out/ --n-clips 5 --vertical --captions
```

`run.sh` just runs `pipeline.py` inside `tf_gpu`. Outputs land in `out/`:

```
out/transcript.json      # word-timestamped transcript
out/clips.json           # chosen clips (start, end, title, reason, hook_score)
out/reels/clip_00_*.mp4  # rendered 1080x1920 reels (+ .srt sidecars)
```

Common flags:

| flag | default | meaning |
|------|---------|---------|
| `-n, --n-clips` | 5 | how many reels to produce |
| `--min-dur` / `--max-dur` | 15 / 60 | clip length bounds (seconds) |
| `--model` | `large-v3` | whisper size (`base`/`small`/`medium`/`large-v3`) |
| `--llm-model` | `claude-opus-4-8` | model for clip selection |
| `--vertical` | off | reframe to 9:16 1080x1920 |
| `--captions` | off | burn word-grouped captions |
| `--no-reuse` | off | force re-transcribe (default reuses `transcript.json`) |

You can also run any stage on its own:

```bash
python -m services.transcribe.transcribe in.mp4 -o transcript.json --model large-v3
python -m services.selector.select_clips transcript.json -o clips.json -n 5
python -m services.cutter.cut in.mp4 clips.json -o reels/ --vertical --captions --transcript transcript.json
```

## Architecture

Services-first. Each stage is self-contained with its own tests; they only
share the typed contracts in [contracts/schemas.py](contracts/schemas.py).

| path | role |
|------|------|
| [contracts/schemas.py](contracts/schemas.py) | `Transcript`, `Clip`, and the LLM output JSON Schema |
| [services/transcribe/](services/transcribe/) | ffmpeg audio extract + faster-whisper GPU transcription |
| [services/selector/](services/selector/) | prompt + local-Claude call + deterministic clip validation |
| [services/cutter/](services/cutter/) | ffmpeg trim, 9:16 reframe, caption burn-in |
| [pipeline.py](pipeline.py) | orchestration glue only |

**Why deterministic validation after the LLM:** the model proposes clips in
latent space (judgment), then [validate_clips](services/selector/select_clips.py)
clamps them in-bounds, enforces length limits, drops overlaps (keeping the
higher hook score), and re-indexes. The cutter never sees a bad window.

## Tests and evals

Two lanes (per project policy):

```bash
# Gate tests — deterministic, no GPU, no LLM, ~3s. Run on every change.
conda run -n tf_gpu python -m pytest -q

# Selector eval — spends one real LLM call, checks the model finds planted
# viral moments. Run before ship / nightly.
conda run -n tf_gpu python -m services.selector.evals.eval_selector
```

The gate suite includes a real-ffmpeg integration test
([tests/test_cut_integration.py](tests/test_cut_integration.py)) that generates
a video and verifies a cut reel is genuinely 1080x1920 with the right duration.

## Reproduce the demo

There's no committed video (binaries stay out of git). To rebuild the sample
talking-head clip used during development:

```bash
conda run -n tf_gpu pip install gTTS
# generate speech, mux onto a test pattern, then run the pipeline
python - <<'PY'
from gtts import gTTS
gTTS("Here is the one habit that doubled my income...", lang="en").save("/tmp/s.mp3")
PY
ffmpeg -y -f lavfi -i "testsrc=size=1280x720:rate=24:duration=92" -i /tmp/s.mp3 \
  -c:v libx264 -preset ultrafast -c:a aac -shortest assets/sample_talk.mp4
./run.sh assets/sample_talk.mp4 -o out_demo --n-clips 3 --model base --vertical --captions
```
