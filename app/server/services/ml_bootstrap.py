# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import os

###############################################################################
def configure_environment() -> None:
    os.environ["KERAS_BACKEND"] = "torch"
    os.environ["MPLBACKEND"] = "Agg"
