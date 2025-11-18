# pylint: disable=missing-module-docstring

import sys
from functools import partial

eprint = partial(print, file=sys.stderr)
