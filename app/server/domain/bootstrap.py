# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from pydantic import BaseModel

###############################################################################
class ServiceStatusResponse(BaseModel):
    status: str
