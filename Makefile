# agentorchestr — dev convenience targets.
#
# `make test`    run the full pytest suite
# `make build`   produce a wheel + sdist in dist/
# `make install` editable install into the active venv
# `make clean`   remove all build / pyc / pytest / venv-build artefacts
# `make smoke`   build wheel + install into a throwaway venv + run --help

PYTHON ?= python3
VENV   ?= .env

.PHONY: test build install clean smoke

test:
	$(VENV)/bin/python -m pytest -q

build: clean
	$(VENV)/bin/python -m build

install:
	$(VENV)/bin/pip install -e ".[dev]"

clean:
	rm -rf build dist *.egg-info agentorchestr.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +

smoke: build
	@rm -rf /tmp/agentorchestr-smoke-venv
	@$(PYTHON) -m venv /tmp/agentorchestr-smoke-venv
	@/tmp/agentorchestr-smoke-venv/bin/pip install --quiet 'mcp>=1.20'
	@/tmp/agentorchestr-smoke-venv/bin/pip install --quiet dist/*.whl
	@echo "--- agentorchestr --help ---"
	@/tmp/agentorchestr-smoke-venv/bin/agentorchestr --help | head -5
	@echo "--- agentorchestr-shim --help ---"
	@/tmp/agentorchestr-smoke-venv/bin/agentorchestr-shim --help | head -5
	@echo "--- agentorchestr --version ---"
	@/tmp/agentorchestr-smoke-venv/bin/agentorchestr --version
	@rm -rf /tmp/agentorchestr-smoke-venv
	@echo "✓ smoke install passed"
