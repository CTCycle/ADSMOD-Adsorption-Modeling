from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


###############################################################################
class NISTFetchRequest(BaseModel):
    dataset_name: str = Field(default="NIST")
    experiments_fraction: float = Field(default=1.0, ge=0.0, le=1.0)
    guest_fraction: float = Field(default=1.0, ge=0.0, le=1.0)
    host_fraction: float = Field(default=1.0, ge=0.0, le=1.0)


###############################################################################
class NISTFetchResponse(BaseModel):
    status: str = Field(default="success")
    dataset_name: str
    experiments_count: int
    single_component_rows: int
    binary_mixture_rows: int
    guest_rows: int
    host_rows: int


###############################################################################
class NISTPropertiesRequest(BaseModel):
    target: Literal["guest", "host"] = Field(default="guest")


###############################################################################
class NISTPropertiesResponse(BaseModel):
    status: str = Field(default="success")
    target: str
    names_requested: int
    names_matched: int
    rows_updated: int


###############################################################################
class NISTStatusResponse(BaseModel):
    status: str = Field(default="success")
    data_available: bool
    single_component_rows: int
    binary_mixture_rows: int
    guest_rows: int
    host_rows: int
