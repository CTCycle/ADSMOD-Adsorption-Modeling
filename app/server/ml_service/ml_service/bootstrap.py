from __future__ import annotations

import os

###############################################################################
def configure_environment() -> None:
    os.environ.setdefault("KERAS_BACKEND", "torch")
    os.environ.setdefault("MPLBACKEND", "Agg")
