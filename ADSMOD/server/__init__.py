from __future__ import annotations

import os

# Force Keras 3 to use the Torch backend unless explicitly overridden.
os.environ.setdefault("KERAS_BACKEND", "torch")
