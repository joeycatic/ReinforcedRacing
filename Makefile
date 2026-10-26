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

train:
	$(UV) run --locked python -m train.ppo $(ARGS)

eval:
	$(UV) run --locked python -m eval.evaluate $(ARGS)

web-dev web-build export:
	@echo "$@ is outside M3; see spec.md milestones M4–M6." >&2
	@exit 2
