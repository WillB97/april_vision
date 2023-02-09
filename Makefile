.PHONY: lint type test isort isort-check build upload

PYMODULE:=april_vision
TESTS:=tests

all: lint isort-check type build

lint:
	flake8 $(PYMODULE)

type:
	mypy $(PYMODULE)

test:
	pytest --cov=$(PYMODULE) $(TESTS)

isort-check:
	python -m isort --check $(PYMODULE)

isort:
	python -m isort $(PYMODULE)

build:
	python -m build

upload:
	twine upload dist/*

clean:
	rm -rf dist/* build/*
