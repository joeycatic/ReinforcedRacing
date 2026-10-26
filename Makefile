UV ?= $(shell command -v uv 2>/dev/null || echo "$(HOME)/.local/bin/uv")
ARGS ?=

.PHONY: setup test drive tracks benchmark train eval export web-dev web-build
setup:
	$(UV) sync --locked

test:
	$(UV) run --locked ruff check .
	$(UV) run --locked ruff format --check .
	$(UV) run --locked pytest

drive:
	$(UV) run --locked python -m tools.debug_drive $(ARGS)

tracks:
	$(UV) run --locked python -m tools.export_tracks $(ARGS)

benchmark:
	$(UV) run --locked python -m tools.benchmark $(ARGS)

train eval export web-dev web-build:
	@echo "$@ is outside M2; see spec.md milestones M3–M6." >&2
	@exit 2
