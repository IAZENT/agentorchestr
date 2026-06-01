# Contributing to agentorchestr

Thanks for taking the time to look.  agentorchestr stays small on purpose — the
whole codebase is under 5 KLOC and you should be able to read it in an
afternoon.  Most contributions land best as small, testable patches.

## Getting set up

```bash
git clone https://github.com/IAZENT/agentorchestr
cd agentorchestr
python3 -m venv .env && source .env/bin/activate
pip install -e ".[dev]"     # core + mcp + zeroconf + sqlite-vec + ddgs + pytest
pytest -q                    # ~120 tests, ~2 s
```

Optional but recommended:

```bash
agentorchestr --detect                # confirms agent + LLM + dependency status
agentorchestr --init                  # if you want to dogfood agentorchestr on this repo
```

## Submitting changes

1. **One change per PR.**  Mixed PRs are slow to review.
2. **Tests come with the code.**  Every behaviour change needs at least
   one test that pins the new contract.  Look at `tests/test_*.py` for
   patterns — most files use `pytest-asyncio` and stub the network with
   `httpx.MockTransport`.
3. **No dead code.**  If you add a module, wire it up.  Unused imports
   are routinely cleaned up.
4. **Match the style.**  4-space indent, type hints on public functions,
   docstrings start with one summary line.  No need for sphinx/docstring
   formatting.
5. **Keep the cacheable prefix stable.**  Adding to
   `SUPERVISOR_PROMPT_TEMPLATE` or trimming MCP tool docstrings has
   real token-cost implications — see `CHANGELOG.md` for the design
   rationale.

## What we welcome most

- New worker perspectives (`perspectives.py`) for specific kinds of
  reviews.
- New skill bundles (under `skills/`) for common stacks.
- Adapters in `agent_detector.py`'s `AGENT_REGISTRY` for additional CLI
  agents.
- Bug reports with reproducible test cases.

## What we usually decline

- New LLM provider integrations that require local hardware (the
  router is hosted-only on purpose; see CHANGELOG 0.3.0).
- Deep refactors that don't carry tests showing the regression they
  prevent.
- Vendor-specific additions that bundle proprietary services.

## Releasing (maintainers)

```bash
# bump pyproject.toml version + add a CHANGELOG entry
git tag v0.3.1 && git push --tags
python -m build              # produces dist/*.whl + dist/*.tar.gz
python -m twine upload dist/* # PyPI
```
