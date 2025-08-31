.PHONY: all clean

VENV_DIR := .venv
PYTHON := python3
MAIN_FILE := train.py

all:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Creating virtual environment..."; \
		python3 -m venv $(VENV_DIR) ; \
		echo "\n=============================================="; \
		echo "Virtual environment created. Please activate it manually:"; \
		echo "\n\tsource $(VENV_DIR)/bin/activate\n"; \
		echo "Then run 'make' again to install dependencies."; \
		echo "=============================================="; \
	else \
		if [ "$(VIRTUAL_ENV)" != "" ] && [ "$(VIRTUAL_ENV)" = "$(shell realpath $(VENV_DIR))" ]; then \
			echo "Virtual environment is already active."; \
			echo "Installing dependencies..."; \
			pip install --upgrade pip ; \
			pip install -r requirements.txt ; \
			echo "\n=============================================="; \
			echo "Dependencies installed. You can now run your script:"; \
			echo "\n\t$(PYTHON) $(MAIN_FILE)\n"; \
			echo "(Do not hesitate to try the --bonus option)"; \
			echo "=============================================="; \
		else \
			echo "Virtual environment exists but is not active."; \
			echo "\n=============================================="; \
			echo "Please activate it manually:"; \
			echo "\n\tsource $(VENV_DIR)/bin/activate\n"; \
			echo "Then run 'make' again to install dependencies."; \
			echo "=============================================="; \
		fi; \
	fi

clean:
	@if [ -d "$(VENV_DIR)" ]; then \
		echo "Are you sure you want to remove the virtual environment '$(VENV_DIR)'?"; \
		echo "Press 'y' to confirm, any other key to cancel."; \
		read -p "> " ans; \
		if [ "$$ans" = "y" ] || [ -z "$$ans" ]; then \
			rm -rf $(VENV_DIR); \
			echo "Virtual environment removed."; \
			echo "Note: You may also want to deactivate the virtual environment if it's active."; \
			echo "To do so, run:\n" ; \
			echo "\t'deactivate'\n" ; \
			echo "in your shell."; \
		else \
			echo "Operation cancelled."; \
		fi; \
	else \
		echo "Virtual environment '$(VENV_DIR)' does not exist. Nothing to remove."; \
	fi

help:
	@echo "Makefile for managing the project's Python environment and scripts."
	@echo ""
	@echo "Usage:"
	@echo "  make all      - Creates the virtual environment if it doesn't exist, and"
	@echo "                  installs project dependencies if the environment is active."
	@echo "  make clean    - Deletes the virtual environment with a confirmation prompt."
	@echo "  make help     - Displays this help message."
	@echo ""
	@echo "Scripts:"
	@echo "  The 'train.py' script must be run before 'predict.py'."
	@echo "  Both have a '--help' option for more details, and a --bonus option for visualisation."
	@echo "  1. Train your model by running 'python3 train.py'."
	@echo "     - This creates the 'model.txt' file."
	@echo ""
	@echo "  2. Use the trained model to predict a car's price with 'predict.py'."
	@echo "     - Run 'python3 predict.py <mileage>' to get a price prediction."
