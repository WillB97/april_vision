.PHONY: lint type test isort build upload

PYMODULE:=april_vision
TESTS:=tests

all: lint type build

lint:
	flake8 $(PYMODULE)

type:
	mypy $(PYMODULE) stubs

test:
	pytest --cov=$(PYMODULE) $(TESTS)

isort:
	python -m isort $(PYMODULE)

build:
	python -m build

upload:
	twine upload dist/*

clean:
	rm -rf dist/* build/*
