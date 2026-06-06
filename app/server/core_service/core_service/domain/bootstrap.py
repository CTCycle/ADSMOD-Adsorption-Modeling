from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock

from pydantic import BaseModel


@dataclass
class EnvironmentBootstrapState:
    lock: Lock = field(default_factory=Lock)
    bootstrapped: bool = False


class ServiceStatusResponse(BaseModel):
    status: str
