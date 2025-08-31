.PHONY: all clean

VENV_DIR := .venv
PYTHON := python3

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
			echo "\n\t$(PYTHON) train.py\n"; \
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
		else \
			echo "Operation cancelled."; \
		fi; \
	else \
		echo "Virtual environment '$(VENV_DIR)' does not exist. Nothing to remove."; \
	fi
