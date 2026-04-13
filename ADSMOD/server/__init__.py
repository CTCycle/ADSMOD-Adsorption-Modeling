from __future__ import annotations

import os

from ADSMOD.server.configurations.environment import load_environment

load_environment()

# Force Keras 3 to use the Torch backend unless explicitly overridden.
os.environ.setdefault("KERAS_BACKEND", "torch")

