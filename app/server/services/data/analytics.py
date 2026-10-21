# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import math
from typing import Any

import numpy as np

from server.common.units import UnitConversionError, UnitRegistry
from server.domain.analytics import (
    MAX_SERIES,
    MAX_SERIES_POINTS,
    AnalyticsFilter,
    AnalyticsOverviewResponse,
    DistributionRequest,
    DistributionResponse,
    DistributionStats,
    HistogramBin,
    IsothermSeries,
    SelectionFacets,
    SelectionRequest,
    SelectionResponse,
    SeriesPoint,
    SeriesRequest,
    SeriesResponse,
    UptakeAtPressureRequest,
    UptakeAtPressureResponse,
    UptakeAtPressureRow,
)
from server.domain.public_data import Pagination
from server.repositories.analytics import AnalyticsRepository

RELATIVE_PRESSURE_UNIT = "1"
EXACT_MATCH_TOLERANCE = 1e-9
OVERVIEW_TEMPERATURE_BINS = 20

###############################################################################
def adsorption_branch(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Points up to the pressure maximum in acquisition order, sorted by pressure.

    Stored observations carry no branch marker, so any desorption points that
    follow the highest pressure are excluded from fixed-pressure comparisons.
    """
    if not points:
        return []
    peak = max(range(len(points)), key=lambda index: points[index]["pressure"])
    return sorted(points[: peak + 1], key=lambda point: point["pressure"])

###############################################################################
def uptake_at_pressure(
    branch: list[dict[str, Any]], target: float, *, method: str, tolerance: float
) -> dict[str, Any]:
    """Measured, nearest or log-pressure interpolated uptake; never extrapolates."""
    if not branch:
        return {"status": "out_of_range"}
    exact = [
        point
        for point in branch
        if math.isclose(point["pressure"], target, rel_tol=EXACT_MATCH_TOLERANCE)
    ]
    if exact:
        return {
            "status": "ok",
            "method": "measured",
            "value": float(np.mean([point["uptake"] for point in exact])),
            "sources": exact,
        }
    if method == "nearest":
        nearest = min(branch, key=lambda point: abs(point["pressure"] - target))
        if abs(nearest["pressure"] - target) / target > tolerance:
            return {"status": "no_point_in_tolerance", "sources": [nearest]}
        return {
            "status": "ok",
            "method": "nearest",
            "value": nearest["uptake"],
            "sources": [nearest],
        }
    if target < branch[0]["pressure"] or target > branch[-1]["pressure"]:
        return {"status": "out_of_range"}
    for lower, upper in zip(branch, branch[1:]):
        if lower["pressure"] < target < upper["pressure"]:
            if lower["pressure"] > 0:
                fraction = math.log(target / lower["pressure"]) / math.log(
                    upper["pressure"] / lower["pressure"]
                )
            else:
                fraction = (target - lower["pressure"]) / (
                    upper["pressure"] - lower["pressure"]
                )
            value = lower["uptake"] + fraction * (upper["uptake"] - lower["uptake"])
            return {
                "status": "ok",
                "method": "interpolated",
                "value": value,
                "sources": [lower, upper],
            }
    return {"status": "out_of_range"}

###############################################################################
def histogram(values: list[float], *, bins: int, scale: str) -> dict[str, Any]:
    data = np.asarray(values, dtype=float)
    excluded = 0
    if scale == "log":
        positive = data > 0
        excluded = int(data.size - positive.sum())
        data = data[positive]
    if data.size == 0:
        return {"bins": [], "stats": DistributionStats(count=0), "excluded": excluded}
    low, high = float(data.min()), float(data.max())
    if scale == "log":
        if low == high:
            low, high = low / 1.1, high * 1.1
        edges = np.geomspace(low, high, bins + 1)
    else:
        if low == high:
            low, high = low - 0.5, high + 0.5
        edges = np.linspace(low, high, bins + 1)
    counts, edges = np.histogram(data, bins=edges)
    return {
        "bins": [
            HistogramBin(lower=float(edges[i]), upper=float(edges[i + 1]), count=int(count))
            for i, count in enumerate(counts)
        ],
        "stats": DistributionStats(
            count=int(data.size),
            min=float(data.min()),
            max=float(data.max()),
            mean=float(data.mean()),
            median=float(np.median(data)),
        ),
        "excluded": excluded,
    }

###############################################################################
class AnalyticsService:
    """Compose read-only analytical views; nothing here is ever persisted."""

    # -------------------------------------------------------------------------
    def __init__(self, repository: AnalyticsRepository) -> None:
        self.repository = repository

    # -------------------------------------------------------------------------
    @staticmethod
    def _filters(analytics_filter: AnalyticsFilter) -> dict[str, Any]:
        return analytics_filter.model_dump()

    # -------------------------------------------------------------------------
    @staticmethod
    def _pressure_factor(basis: str | None, unit: str) -> tuple[str, float]:
        """Display unit and divisor from canonical pressure."""
        if basis == "relative":
            return RELATIVE_PRESSURE_UNIT, 1.0
        resolved = UnitRegistry.pressure_unit(unit)
        if resolved not in UnitRegistry.PRESSURE_TO_PA:
            raise UnitConversionError(
                "Absolute and partial pressures need a dimensional display unit."
            )
        return resolved, UnitRegistry.PRESSURE_TO_PA[resolved]

    # -------------------------------------------------------------------------
    @staticmethod
    def _uptake_factor(
        unit: str, molar_mass: float | None, warnings: list[str], adsorbate: str
    ) -> tuple[str, float]:
        """Display unit and divisor from mol/kg, falling back to mol/kg."""
        try:
            resolved, factor, _ = UnitRegistry.uptake_factor_to_mol_kg(unit, molar_mass)
        except UnitConversionError as exc:
            message = f"{adsorbate}: {exc} Shown in mol/kg."
            if message not in warnings:
                warnings.append(message)
            return "mol/kg", 1.0
        return resolved, factor

    # -------------------------------------------------------------------------
    @staticmethod
    def _single_basis(rows: list[dict[str, Any]]) -> str | None:
        bases = sorted({row["pressure_basis"] for row in rows})
        if len(bases) > 1:
            raise ValueError(
                "The selection mixes pressure bases ("
                + ", ".join(bases)
                + "); choose one pressure basis to share a pressure axis."
            )
        return bases[0] if bases else None

    # -------------------------------------------------------------------------
    def _series_rows(
        self, analytics_filter: AnalyticsFilter
    ) -> tuple[list[dict[str, Any]], int, str | None]:
        rows, total = self.repository.series_metadata(
            self._filters(analytics_filter), limit=MAX_SERIES
        )
        basis = self._single_basis(rows)
        kept: list[dict[str, Any]] = []
        points = 0
        for row in rows:
            points += int(row["point_count"])
            if points > MAX_SERIES_POINTS and kept:
                break
            kept.append(row)
        return kept, total, basis

    # -------------------------------------------------------------------------
    @staticmethod
    def _metadata(row: dict[str, Any], uptake_unit: str) -> dict[str, Any]:
        return {
            key: row[key]
            for key in (
                "isotherm_id",
                "component_id",
                "isotherm",
                "dataset_id",
                "dataset",
                "dataset_source",
                "material_id",
                "material",
                "adsorbate_id",
                "adsorbate",
                "mole_fraction",
                "temperature_k",
            )
        } | {"uptake_unit": uptake_unit}

    # -------------------------------------------------------------------------
    def selection(self, request: SelectionRequest) -> SelectionResponse:
        filters = self._filters(request.filter)
        items, total = self.repository.selection_page(
            filters, page=request.page, page_size=request.page_size
        )
        facets = self.repository.facets(
            filters,
            material_search=request.material_search,
            adsorbate_search=request.adsorbate_search,
        )
        return SelectionResponse.model_validate(
            {
                "items": items,
                "pagination": Pagination(
                    page=request.page, page_size=request.page_size, total=total
                ),
                "facets": SelectionFacets.model_validate(facets),
            }
        )

    # -------------------------------------------------------------------------
    def series(self, request: SeriesRequest) -> SeriesResponse:
        rows, total, basis = self._series_rows(request.filter)
        pressure_unit, pressure_divisor = self._pressure_factor(basis, request.pressure_unit)
        observations = self.repository.component_observations(
            [int(row["component_id"]) for row in rows]
        )
        warnings: list[str] = []
        series = []
        for row in rows:
            uptake_unit, uptake_factor = self._uptake_factor(
                request.uptake_unit, row["molar_mass_g_mol"], warnings, row["adsorbate"]
            )
            series.append(
                IsothermSeries(
                    **self._metadata(row, uptake_unit),
                    points=[
                        SeriesPoint(
                            observation_id=point["observation_id"],
                            pressure=point["pressure"] / pressure_divisor,
                            uptake=point["uptake"] / uptake_factor,
                            uptake_stddev=(
                                None
                                if point["uptake_stddev"] is None
                                else point["uptake_stddev"] / uptake_factor
                            ),
                        )
                        for point in observations.get(int(row["component_id"]), [])
                    ],
                )
            )
        return SeriesResponse(
            pressure_basis=basis,  # type: ignore[arg-type]
            pressure_unit=pressure_unit,
            series=series,
            total_series=total,
            truncated=len(series) < total,
            warnings=warnings,
        )

    # -------------------------------------------------------------------------
    def uptake_at_pressure(self, request: UptakeAtPressureRequest) -> UptakeAtPressureResponse:
        rows, total, basis = self._series_rows(request.filter)
        pressure_unit, pressure_divisor = self._pressure_factor(basis, request.pressure_unit)
        target = request.pressure * pressure_divisor
        observations = self.repository.component_observations(
            [int(row["component_id"]) for row in rows]
        )
        warnings: list[str] = []
        results = []
        for row in rows:
            uptake_unit, uptake_factor = self._uptake_factor(
                request.uptake_unit, row["molar_mass_g_mol"], warnings, row["adsorbate"]
            )
            outcome = uptake_at_pressure(
                adsorption_branch(observations.get(int(row["component_id"]), [])),
                target,
                method=request.method,
                tolerance=request.tolerance,
            )
            value = outcome.get("value")
            results.append(
                UptakeAtPressureRow(
                    **self._metadata(row, uptake_unit),
                    value=None if value is None else value / uptake_factor,
                    method=outcome.get("method"),
                    status=outcome["status"],
                    sources=[
                        SeriesPoint(
                            observation_id=point["observation_id"],
                            pressure=point["pressure"] / pressure_divisor,
                            uptake=point["uptake"] / uptake_factor,
                        )
                        for point in outcome.get("sources", [])
                    ],
                )
            )
        return UptakeAtPressureResponse(
            pressure=request.pressure,
            pressure_unit=pressure_unit,
            pressure_basis=basis,  # type: ignore[arg-type]
            method=request.method,
            rows=results,
            total_series=total,
            truncated=len(results) < total,
            warnings=warnings,
        )

    # -------------------------------------------------------------------------
    def distribution(self, request: DistributionRequest) -> DistributionResponse:
        values = self.repository.distribution_values(
            self._filters(request.filter), request.variable
        )
        unit = {"temperature_k": "K", "uptake_mol_kg": "mol/kg"}.get(
            request.variable, "points"
        )
        if request.variable == "pressure":
            basis = request.filter.pressure_basis
            if basis is None:
                raise ValueError("A pressure distribution requires a pressure basis.")
            unit, divisor = self._pressure_factor(basis, request.pressure_unit)
            values = [value / divisor for value in values]
        result = histogram(values, bins=request.bins, scale=request.scale)
        return DistributionResponse(
            variable=request.variable, unit=unit, scale=request.scale, **result
        )

    # -------------------------------------------------------------------------
    def overview(self) -> AnalyticsOverviewResponse:
        summary = self.repository.overview()
        temperatures = self.repository.distribution_values(
            {"include_mixtures": True}, "temperature_k"
        )
        return AnalyticsOverviewResponse.model_validate(
            summary
            | {
                "temperature_histogram": histogram(
                    temperatures, bins=OVERVIEW_TEMPERATURE_BINS, scale="linear"
                )["bins"]
            }
        )


__all__ = ["AnalyticsService", "adsorption_branch", "histogram", "uptake_at_pressure"]
