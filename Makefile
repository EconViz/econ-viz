UV ?= uv
PYTHON := $(UV) run python

.PHONY: sync examples example-ic example-eq example-themes example-latex clean test build

sync:
	$(UV) sync --frozen --all-extras

examples: example-ic example-eq example-themes example-latex

example-ic:
	$(PYTHON) -m examples.indifference_curves

example-eq:
	$(PYTHON) -m examples.equilibrium

example-themes:
	$(PYTHON) -m examples.themes

example-latex:
	$(PYTHON) -m examples.latex_input

clean:
	rm -rf examples/output/*.png

test:
	$(UV) run pytest

build:
	$(UV) build
