"""LLM client: shells out to the local Claude Code binary.

Per project policy we never call a hosted LLM API. We invoke the user's local
`claude` CLI in headless mode and ask for structured output:

    claude -p <prompt> --output-format json --json-schema <schema> --model <m>

`--output-format json` wraps the reply in an envelope; `--json-schema`
constrains the model's `result` to our shape. We parse the envelope, then the
result, and hand back a plain dict.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
from typing import Any

DEFAULT_MODEL = "claude-opus-4-8"  # best available by default, per policy

# Known locations for the local binary, in priority order. PATH wins if present.
_EXTENSION_GLOB = os.path.expanduser(
    "~/.vscode/extensions/anthropic.claude-code-*/resources/native-binary/claude"
)


def find_claude_bin() -> str:
    """Locate the local claude binary. Raises if it can't be found."""
    env = os.environ.get("CLAUDE_BIN")
    if env and os.path.isfile(env):
        return env

    from shutil import which
    on_path = which("claude")
    if on_path:
        return on_path

    matches = sorted(glob.glob(_EXTENSION_GLOB))
    if matches:
        return matches[-1]  # highest version

    raise RuntimeError(
        "Could not find the local `claude` binary. Set CLAUDE_BIN or add claude to PATH."
    )


def _extract_result(envelope_text: str) -> Any:
    """Pull the model's structured payload out of the CLI envelope.

    With --output-format json stdout is a JSON envelope. When --json-schema is
    used, the validated object lives in `structured_output` (the `result` field
    holds the model's plain-text reply instead). Without a schema, the payload
    is in `result`, possibly as a JSON string we must parse. We handle both,
    plus the degenerate case where stdout is already the payload.
    """
    text = envelope_text.strip()
    try:
        envelope = json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"claude returned non-JSON output: {text[:500]!r}") from e

    if isinstance(envelope, dict) and envelope.get("is_error"):
        raise RuntimeError(f"claude reported an error: {envelope.get('result')!r}")

    # Schema-constrained output is delivered here, already a dict.
    if isinstance(envelope, dict) and envelope.get("structured_output") is not None:
        return envelope["structured_output"]

    if isinstance(envelope, dict) and "result" in envelope:
        result = envelope["result"]
    else:
        result = envelope

    if isinstance(result, str):
        result = result.strip()
        try:
            result = json.loads(result)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"model result was not valid JSON: {result[:500]!r}") from e
    return result


def build_claude_cmd(
    claude_bin: str,
    prompt: str,
    schema: dict[str, Any] | None,
    model: str,
) -> list[str]:
    """Assemble the headless claude command (pure, so it is gate-testable)."""
    cmd = [claude_bin, "-p", prompt, "--output-format", "json", "--model", model]
    if schema is not None:
        cmd += ["--json-schema", json.dumps(schema)]
    return cmd


def call_claude(
    prompt: str,
    schema: dict[str, Any] | None = None,
    model: str = DEFAULT_MODEL,
    claude_bin: str | None = None,
    timeout: int = 300,
) -> Any:
    """Run the local claude headless and return the parsed structured result."""
    claude_bin = claude_bin or find_claude_bin()
    cmd = build_claude_cmd(claude_bin, prompt, schema, model)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"claude exited {proc.returncode}: {proc.stderr.strip()[:500]}"
        )
    return _extract_result(proc.stdout)
