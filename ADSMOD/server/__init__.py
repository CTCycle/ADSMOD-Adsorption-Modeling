from __future__ import annotations

import numpy as np

# Compatibility shim for Keras/tf2onnx expecting np.object on NumPy >= 2.0.
if not hasattr(np, "object"):
    np.object = np.object_

