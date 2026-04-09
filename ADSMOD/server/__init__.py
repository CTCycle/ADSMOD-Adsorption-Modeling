from __future__ import annotations

import os

from ADSMOD.server.configurations.bootstrap import ensure_environment_loaded

ensure_environment_loaded()

# Force Keras 3 to use the Torch backend unless explicitly overridden.
os.environ.setdefault("KERAS_BACKEND", "torch")
