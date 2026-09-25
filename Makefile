UV ?= uv
PYTHON := $(UV) run python
EXAMPLE_ENV := MPLBACKEND=Agg
EXAMPLE_SCRIPTS := $(sort $(wildcard examples/scripts/*.py))
STATIC_EXAMPLE_SCRIPTS := $(filter-out examples/scripts/animation.py,$(EXAMPLE_SCRIPTS))

.PHONY: sync examples examples-static examples-smoke example-animation-smoke example-ic example-eq example-themes example-latex clean test build

sync:
	$(UV) sync --frozen --all-extras

examples:
	@set -e; for script in $(EXAMPLE_SCRIPTS); do \
		echo "Running $$script"; \
		$(EXAMPLE_ENV) $(PYTHON) $$script; \
	done

examples-static:
	@set -e; for script in $(STATIC_EXAMPLE_SCRIPTS); do \
		echo "Running $$script"; \
		$(EXAMPLE_ENV) $(PYTHON) $$script; \
	done

example-animation-smoke:
	$(EXAMPLE_ENV) $(PYTHON) examples/scripts/animation.py --smoke

examples-smoke: clean examples-static example-animation-smoke

example-ic:
	$(PYTHON) examples/scripts/indifference_curves.py

example-eq:
	$(PYTHON) examples/scripts/equilibrium.py

example-themes:
	$(PYTHON) examples/scripts/themes.py

example-latex:
	$(PYTHON) examples/scripts/latex_input.py

clean:
	rm -rf examples/output/*

test:
	$(UV) run pytest

build:
	$(UV) build
