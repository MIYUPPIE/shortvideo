# selector service

Transcript in, clean list of `Clip`s out. Two halves:

**Latent (LLM).** `llm_client.call_claude` shells out to the local `claude` CLI:

```
claude -p <prompt> --output-format json --json-schema <schema> --model <m>
```

The schema-validated object comes back in the envelope's `structured_output`
field (not `result`, which holds the model's plain text). `prompt.build_prompt`
formats the timestamped transcript and the editing rules.

**Deterministic (code).** `validate_clips` is the safety net over the model:
clamp to `[0, duration]`, drop clips under `min_dur`, trim clips over `max_dur`,
greedily remove overlaps keeping the higher `hook_score`, sort by start,
re-index. This is what the gate tests cover.

```bash
python -m services.selector.select_clips transcript.json -o clips.json -n 5
```

- Contract in: [Transcript](../../contracts/schemas.py); out: list of `Clip`.
- LLM output schema: `CLIP_SELECTION_SCHEMA` in the same file.
- Default model: `claude-opus-4-8`. Override the binary with `$CLAUDE_BIN`.
- Tests: `tests/`. Eval (real LLM call): `evals/eval_selector.py`.
