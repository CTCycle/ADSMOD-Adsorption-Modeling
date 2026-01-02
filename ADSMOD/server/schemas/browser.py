from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


###############################################################################
class TableInfo(BaseModel):
    table_name: str
    display_name: str
    category: str


class TableListResponse(BaseModel):
    status: str = Field(default="success")
    tables: list[TableInfo]


class TableDataResponse(BaseModel):
    status: str = Field(default="success")
    table_name: str
    display_name: str
    row_count: int
    column_count: int
    columns: list[str]
    data: list[dict[str, Any]]
