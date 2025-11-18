"""
Short file to store an utils like function : `ensure_env`.
"""


import os
import sys
from .utils import eprint


def ensure_env():
    """
    Prevents execution if the environment is not venv.
    """
    expected = os.path.join(os.getcwd(), ".venv", "bin", "python")
    current = sys.executable
    if current != expected:
        eprint(
            "This script must be run inside the project virtual environment!\n"
            f"Expected: {expected}\n"
            f"Found: {current}\n"
            "Make sure to run the python files with the run.sh"
            "script (which handles environment issues):\n\n"
            "\t./run.sh train [args...]\n"
            "Or\n"
            "\t./run.sh predict [args...]\n"
        )
        sys.exit(1)
