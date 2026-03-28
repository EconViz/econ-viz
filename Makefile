PYTHON := $(shell if [ -n "$$VIRTUAL_ENV" ]; then echo python3; else echo poetry run python3; fi)

.PHONY: examples example-ic example-eq example-themes example-latex clean test

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
	$(PYTHON) -m pytest tests/
