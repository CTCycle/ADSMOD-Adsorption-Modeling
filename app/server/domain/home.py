# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from pydantic import BaseModel


###############################################################################
class HomeSourceCount(BaseModel):
    key: str
    name: str
    record_count: int


###############################################################################
class HomeSummaryResponse(BaseModel):
    """Counts of persisted entities.

    Species and materials are stored entity rows, the same population the
    Datasets page lists, whether they came from an upload or a provider.
    """

    datasets: int
    experiments: int
    species: int
    materials: int
    sources: list[HomeSourceCount]
    fitting_sessions: int
    fitting_sessions_by_status: dict[str, int]
    latest_fitting_at: str | None = None


__all__ = ["HomeSourceCount", "HomeSummaryResponse"]
