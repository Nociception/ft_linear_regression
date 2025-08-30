.PHONY: all

VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip


create-venv:
	python3 -m venv $(VENV_DIR)
	@echo "Virtual environment created at $(VENV_DIR)."
	@echo "Type source .venv/bin/activate to activate the virtual environment."

install-requirements:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

