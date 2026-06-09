"""Gate tests for the local-Claude client: command build + envelope parsing."""
import json

import pytest

from services.selector.llm_client import (
    build_claude_cmd,
    _extract_result,
)


def test_build_cmd_headless_json_with_schema():
    schema = {"type": "object"}
    cmd = build_claude_cmd("/bin/claude", "hi", schema, "claude-opus-4-8")
    assert cmd[0] == "/bin/claude"
    assert "-p" in cmd and "hi" in cmd
    assert cmd[cmd.index("--output-format") + 1] == "json"
    assert cmd[cmd.index("--model") + 1] == "claude-opus-4-8"
    assert json.loads(cmd[cmd.index("--json-schema") + 1]) == schema


def test_build_cmd_without_schema_omits_flag():
    cmd = build_claude_cmd("/bin/claude", "hi", None, "m")
    assert "--json-schema" not in cmd


def test_extract_result_prefers_structured_output():
    # When --json-schema is used, the validated object is in structured_output;
    # `result` holds the model's text and must be ignored.
    env = json.dumps({
        "result": "Here are your clips!",
        "structured_output": {"clips": [{"start": 16, "end": 33}]},
        "is_error": False,
    })
    assert _extract_result(env) == {"clips": [{"start": 16, "end": 33}]}


def test_extract_result_from_envelope_with_json_string():
    env = json.dumps({"result": json.dumps({"clips": [{"start": 1}]}), "is_error": False})
    assert _extract_result(env) == {"clips": [{"start": 1}]}


def test_extract_result_from_envelope_with_dict_result():
    env = json.dumps({"result": {"clips": []}, "is_error": False})
    assert _extract_result(env) == {"clips": []}


def test_extract_result_bare_payload():
    assert _extract_result(json.dumps({"clips": [1, 2]})) == {"clips": [1, 2]}


def test_extract_result_raises_on_error_envelope():
    env = json.dumps({"result": "boom", "is_error": True})
    with pytest.raises(RuntimeError):
        _extract_result(env)


def test_extract_result_raises_on_garbage():
    with pytest.raises(RuntimeError):
        _extract_result("not json at all")
