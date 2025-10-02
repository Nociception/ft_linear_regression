.PHONY: clean help

VENV_DIR := .venv

clean:
	@if [ -d "$(VENV_DIR)" ]; then \
		echo "Removing virtual environment..."; \
		rm -rf $(VENV_DIR); \
		echo "Virtual environment removed."; \
	else \
		echo "No virtual environment to remove."; \
	fi

help:
	@echo "Usage:"
	@echo "  ./run.sh train [args...]   - Run train.py with optional arguments"
	@echo "  ./run.sh predict [args...] - Run predict.py with optional arguments"
	@echo "  make clean                 - Remove the virtual environment"
