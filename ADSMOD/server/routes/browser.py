from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from ADSMOD.server.schemas.browser import TableDataResponse, TableInfo, TableListResponse
from ADSMOD.server.utils.repository.serializer import DataSerializer
from ADSMOD.server.utils.constants import (
    BROWSER_DATA_ENDPOINT,
    BROWSER_ROUTER_PREFIX,
    BROWSER_TABLE_CATEGORIES,
    BROWSER_TABLE_DISPLAY_NAMES,
    BROWSER_TABLES_ENDPOINT,
)

from ADSMOD.server.utils.logger import logger

serializer = DataSerializer()
router = APIRouter(prefix=BROWSER_ROUTER_PREFIX, tags=["browser"])


###############################################################################
@router.get(
    BROWSER_TABLES_ENDPOINT,
    response_model=TableListResponse,
    status_code=status.HTTP_200_OK,
)
async def list_tables() -> TableListResponse:
    """Return list of available database tables with friendly display names."""
    tables = [
        TableInfo(
            table_name=table_name,
            display_name=display_name,
            category=BROWSER_TABLE_CATEGORIES[table_name],
        )
        for table_name, display_name in BROWSER_TABLE_DISPLAY_NAMES.items()
    ]
    return TableListResponse(tables=tables)


###############################################################################
@router.get(
    f"{BROWSER_DATA_ENDPOINT}/{{table_name}}",
    response_model=TableDataResponse,
    status_code=status.HTTP_200_OK,
)
async def get_table_data(table_name: str) -> TableDataResponse:
    """Fetch all data from the specified table with row and column counts."""
    if table_name not in BROWSER_TABLE_DISPLAY_NAMES:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Table '{table_name}' not found or not available for browsing.",
        )

    try:
        df = serializer.load_table(table_name)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load table %s", table_name)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load table data: {exc}",
        ) from exc

    # Convert DataFrame to list of dicts for JSON response
    columns = df.columns.tolist()
    data = df.fillna("").to_dict(orient="records")

    return TableDataResponse(
        table_name=table_name,
        display_name=BROWSER_TABLE_DISPLAY_NAMES[table_name],
        row_count=len(df),
        column_count=len(columns),
        columns=columns,
        data=data,
    )
