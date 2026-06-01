"""Tests for the perspective system-prompt builder."""
import pytest

from agentorchestr import perspectives


def test_known_perspectives_are_valid():
    names = perspectives.perspective_names()
    assert {"implementer", "tester", "reviewer", "security", "performance",
            "verifier", "researcher"} <= set(names)


def test_invalid_perspective_raises():
    with pytest.raises(ValueError):
        perspectives.system_prompt("debugger", task="x")


def test_prompt_contains_role_marker():
    p = perspectives.system_prompt("implementer", task="add foo()")
    assert "Role: IMPLEMENTER" in p
    assert "add foo()" in p


def test_prompt_includes_sentinels():
    """Every worker prompt must teach the WORKER_DONE / WORKER_BLOCKED /
    WORKER_FAILED sentinels — the supervisor relies on them."""
    p = perspectives.system_prompt("tester", task="add tests")
    assert "WORKER_DONE" in p
    assert "WORKER_BLOCKED" in p
    assert "WORKER_FAILED" in p


def test_file_scope_is_truncated():
    scope = [f"src/file_{i}.py" for i in range(40)]
    p = perspectives.system_prompt("implementer", task="x", file_scope=scope)
    assert "src/file_0.py" in p
    assert "src/file_19.py" in p
    assert "+20 more" in p
    # Files past 20 should NOT appear (token-conscious truncation)
    assert "src/file_25.py" not in p


def test_pass_criteria_truncated_long():
    p = perspectives.system_prompt(
        "implementer", task="x", pass_criteria="A" * 500,
    )
    # Count only the run of literal As we injected; criteria is capped at 400 chars.
    import re
    longest_run = max((len(m.group(0)) for m in re.finditer(r"A+", p)), default=0)
    assert longest_run == 400


def test_verifier_demands_full_test_run():
    """Anthropic's 'no shortcuts' instruction MUST be present in the
    verifier prompt — it's the entire point of the perspective."""
    p = perspectives.system_prompt("verifier", task="run tests")
    assert "FULL test suite" in p
    assert "NEVER take shortcuts" in p


def test_reviewer_is_readonly_by_default():
    assert perspectives.READONLY_HINTS["reviewer"] == ["*"]
    assert perspectives.READONLY_HINTS["implementer"] == []


def test_researcher_prompt_advertises_web_research_tool():
    """The researcher must know about agentorchestr's MCP web_research tool —
    that's the cache-aware path. Native agent web tools are a fallback."""
    p = perspectives.system_prompt("researcher", task="find latest fastapi version")
    assert "Role: RESEARCHER" in p
    assert "web_research" in p
    assert "WORKER_DONE" in p


def test_researcher_is_readonly():
    assert perspectives.READONLY_HINTS["researcher"] == ["*"]
