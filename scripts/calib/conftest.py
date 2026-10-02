"""Make the calibration tools and test helpers importable (tests run with --import-mode=importlib)."""

import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
