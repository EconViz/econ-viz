PYTHON := $(shell if [ -n "$$VIRTUAL_ENV" ]; then echo python3; else echo poetry run python3; fi)

.PHONY: examples example-ic example-eq example-themes clean test

examples: example-ic example-eq example-themes

example-ic:
	$(PYTHON) -m examples.indifference_curves

example-eq:
	$(PYTHON) -m examples.equilibrium

example-themes:
	$(PYTHON) -m examples.themes

clean:
	rm -rf examples/output/*.png

test:
	$(PYTHON) -m pytest tests/
