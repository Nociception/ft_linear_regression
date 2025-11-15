from .utils import eprint
import os
import sys


def ensure_env():
    EXPECTED = os.path.join(os.getcwd(), ".venv", "bin", "python")
    current = sys.executable
    if current != EXPECTED:
        eprint(
            "This script must be run inside the project virtual environment!\n"
            f"Expected: {EXPECTED}\n"
            f"Found: {current}\n"
            "Make sure to run the python files with the run.sh"
            "script (which handles environment issues):\n\n"
            "\t./run.sh train [args...]\n"
            "Or\n"
            "\t./run.sh predict [args...]\n"
        )
        sys.exit(1)
