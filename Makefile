UV ?= $(shell command -v uv 2>/dev/null || echo "$(HOME)/.local/bin/uv")
ARGS ?=

.PHONY: setup test drive tracks benchmark train eval export web-dev web-build parity
setup:
	$(UV) sync --locked
	npm ci --prefix web

test:
	$(UV) run --locked ruff check .
	$(UV) run --locked ruff format --check .
	$(UV) run --locked pytest
	npm run --prefix web typecheck
	npm test --prefix web

drive:
	$(UV) run --locked python -m tools.debug_drive $(ARGS)

tracks:
	$(UV) run --locked python -m tools.export_tracks $(ARGS)

benchmark:
	$(UV) run --locked python -m tools.benchmark $(ARGS)

train:
	$(UV) run --locked python -m train.ppo $(ARGS)

eval:
	$(UV) run --locked python -m eval.artifacts $(ARGS)

export:
	$(UV) run --locked python -m export.assets $(ARGS)

parity:
	$(UV) run --locked python -m tools.generate_parity $(ARGS)

web-dev:
	npm run --prefix web dev -- $(ARGS)

web-build:
	npm run --prefix web build -- $(ARGS)

.PHONY: medals
medals:
	$(UV) run --locked python -m tools.select_medals $(ARGS)

.PHONY: attempts
attempts:
	$(UV) run --locked python -m tools.select_attempts $(ARGS)
