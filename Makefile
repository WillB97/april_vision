.PHONY: lint type test isort isort-check build upload

PYMODULE:=april_vision
TESTS:=tests

all: lint isort-check type test

lint:
	ruff check $(PYMODULE)

type:
	mypy $(PYMODULE)

test:
	pytest --cov=$(PYMODULE) --cov-report=term --cov-report=xml $(TESTS)

fix:
	ruff check --fix-only $(PYMODULE)

build:
	python -m build

upload:
	twine upload dist/*

clean:
	rm -rf dist/* build/*
