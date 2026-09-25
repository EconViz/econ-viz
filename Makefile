UV ?= uv
PYTHON := $(UV) run python
EXAMPLE_SCRIPTS := $(sort $(wildcard examples/scripts/*.py))

.PHONY: sync examples example-ic example-eq example-themes example-latex clean test build

sync:
	$(UV) sync --frozen --all-extras

examples:
	@set -e; for script in $(EXAMPLE_SCRIPTS); do \
		echo "Running $$script"; \
		$(PYTHON) $$script; \
	done

example-ic:
	$(PYTHON) examples/scripts/indifference_curves.py

example-eq:
	$(PYTHON) examples/scripts/equilibrium.py

example-themes:
	$(PYTHON) examples/scripts/themes.py

example-latex:
	$(PYTHON) examples/scripts/latex_input.py

clean:
	rm -rf examples/output

test:
	$(UV) run pytest

build:
	$(UV) build
