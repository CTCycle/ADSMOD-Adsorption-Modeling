# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import math

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import event

from server.api.analytics import AnalyticsEndpoint
from server.configurations.settings import DatabaseConfig
from server.domain.analytics import (
    AnalyticsFilter,
    DistributionRequest,
    SelectionRequest,
    SeriesRequest,
    UptakeAtPressureRequest,
)
from server.repositories.analytics import AnalyticsRepository
from server.repositories.database.manager import DatabaseManager
from server.repositories.schemas import Base
from server.repositories.schemas.models import (
    Adsorbate,
    Adsorbent,
    Dataset,
    Isotherm,
    IsothermComponent,
    Observation,
)
from server.services.data import analytics as analytics_module
from server.services.data.analytics import (
    AnalyticsService,
    adsorption_branch,
    histogram,
    uptake_at_pressure,
)

###############################################################################
def _service(sqlite_path: str = ":memory:") -> tuple[AnalyticsService, dict[str, int]]:
    manager = DatabaseManager(
        DatabaseConfig(embedded_database=True, connect_timeout=5, sqlite_path=sqlite_path)
    )
    Base.metadata.create_all(manager.engine)
    ids: dict[str, int] = {}
    with manager.transaction() as session:
        nist = Dataset(name="NIST", source="nist")
        uploaded = Dataset(name="Lab", source="uploaded")
        material_a = Adsorbent(key="mat-a", name="Material A")
        material_b = Adsorbent(key="mat-b", name="Material B")
        co2 = Adsorbate(key="co2", name="carbon dioxide", molar_mass_g_mol=44.01)
        n2 = Adsorbate(key="n2", name="nitrogen")
        session.add_all([nist, uploaded, material_a, material_b, co2, n2])
        session.flush()
        ids.update(material_a=material_a.id, material_b=material_b.id, co2=co2.id, n2=n2.id)

        def add_isotherm(
            key: str,
            dataset: Dataset,
            material: Adsorbent,
            temperature_k: float,
            basis: str,
            components: list[tuple[Adsorbate, list[tuple[float, float]]]],
        ) -> None:
            isotherm = Isotherm(
                dataset_id=dataset.id,
                external_key=key,
                name=key,
                adsorbent_id=material.id,
                temperature_original=temperature_k,
                temperature_original_unit="K",
                temperature_k=temperature_k,
                pressure_basis=basis,
            )
            session.add(isotherm)
            session.flush()
            ids[key] = isotherm.id
            for position, (adsorbate, points) in enumerate(components, start=1):
                component = IsothermComponent(
                    isotherm_id=isotherm.id, position=position, adsorbate_id=adsorbate.id
                )
                session.add(component)
                session.flush()
                for index, (pressure, uptake) in enumerate(points):
                    session.add(
                        Observation(
                            isotherm_id=isotherm.id,
                            component_id=component.id,
                            sequence_index=index,
                            pressure_original=pressure,
                            pressure_original_unit="Pa" if basis != "relative" else "1",
                            pressure_canonical=pressure,
                            pressure_canonical_unit="Pa" if basis != "relative" else "1",
                            uptake_original=uptake,
                            uptake_original_unit="mol/kg",
                            uptake_mol_kg=uptake,
                        )
                    )

        add_isotherm(
            "a-co2-273", nist, material_a, 273.15, "absolute",
            [(co2, [(1e4, 1.0), (5e4, 2.0), (1e5, 3.0), (2e5, 4.0), (1e5, 3.5)])],
        )
        add_isotherm(
            "a-co2-298", uploaded, material_a, 298.15, "absolute",
            [(co2, [(2e4, 1.0), (3e5, 2.0)])],
        )
        add_isotherm(
            "b-n2-77", uploaded, material_b, 77.0, "relative",
            [(n2, [(0.1, 5.0), (0.5, 8.0)])],
        )
        add_isotherm(
            "a-mix-298", nist, material_a, 298.15, "absolute",
            [(co2, [(1e5, 1.0)]), (n2, [(1e5, 0.2)])],
        )
    return AnalyticsService(AnalyticsRepository(manager)), ids

###############################################################################
def _count_selects(service: AnalyticsService, operation) -> int:  # type: ignore[no-untyped-def]
    engine = service.repository.database.engine
    statements: list[str] = []

    def capture(conn, cursor, statement, parameters, context, executemany) -> None:  # type: ignore[no-untyped-def]
        del conn, cursor, parameters, context, executemany
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append(statement)

    event.listen(engine, "before_cursor_execute", capture)
    try:
        operation()
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    return len(statements)

###############################################################################
def test_selection_facets_keep_sibling_options_visible() -> None:
    service, ids = _service()

    response = service.selection(
        SelectionRequest(filter=AnalyticsFilter(adsorbent_ids=[ids["material_a"]]))
    )

    # Mixtures are excluded by default, so only the two pure CO2 isotherms match.
    assert response.pagination.total == 2
    assert {item.name for item in response.items} == {"a-co2-273", "a-co2-298"}
    assert [(facet.name, facet.count) for facet in response.facets.materials] == [
        ("Material A", 2),
        ("Material B", 1),
    ]
    selected_b = service.selection(
        SelectionRequest(filter=AnalyticsFilter(adsorbent_ids=[ids["material_b"]]))
    )
    assert selected_b.facets.materials[0].name == "Material B"
    assert {facet.name: facet.count for facet in response.facets.adsorbates} == {
        "carbon dioxide": 2
    }
    assert response.facets.sources == {"nist": 1, "uploaded": 1}
    assert response.facets.temperature_min_k == pytest.approx(273.15)

###############################################################################
def test_series_never_mix_pressure_bases_and_convert_display_units() -> None:
    service, ids = _service()

    with pytest.raises(ValueError, match="mixes pressure bases"):
        service.series(SeriesRequest())

    response = service.series(
        SeriesRequest(
            filter=AnalyticsFilter(pressure_basis="absolute", include_mixtures=True),
            pressure_unit="bar",
            uptake_unit="mg/g",
        )
    )

    assert response.pressure_basis == "absolute"
    assert response.pressure_unit == "bar"
    assert response.total_series == 4
    first = next(series for series in response.series if series.isotherm == "a-co2-273")
    assert first.uptake_unit == "mg/g"
    assert first.points[0].pressure == pytest.approx(0.1)
    assert first.points[0].uptake == pytest.approx(44.01)
    nitrogen = next(series for series in response.series if series.adsorbate == "nitrogen")
    assert nitrogen.uptake_unit == "mol/kg"
    assert any("nitrogen" in warning for warning in response.warnings)

###############################################################################
def test_series_respect_point_cap_and_report_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _ = _service()
    monkeypatch.setattr(analytics_module, "MAX_SERIES_POINTS", 5)
    request = SeriesRequest(filter=AnalyticsFilter(pressure_basis="absolute"))

    response = service.series(request)

    assert [series.isotherm for series in response.series] == ["a-co2-273"]
    assert response.total_series == 2
    assert response.truncated is True
    # Count, metadata page and one observation batch, independent of series count.
    assert _count_selects(service, lambda: service.series(request)) == 3

###############################################################################
def test_uptake_at_pressure_interpolates_in_log_pressure_without_extrapolating() -> None:
    points = [
        {"observation_id": 1, "pressure": 1e4, "uptake": 1.0},
        {"observation_id": 2, "pressure": 1e5, "uptake": 2.0},
        {"observation_id": 3, "pressure": 1e6, "uptake": 3.0},
        {"observation_id": 4, "pressure": 5e5, "uptake": 2.9},
    ]
    branch = adsorption_branch(points)

    assert [point["observation_id"] for point in branch] == [1, 2, 3]
    measured = uptake_at_pressure(branch, 1e5, method="linear", tolerance=0.05)
    assert (measured["method"], measured["value"]) == ("measured", 2.0)
    midpoint = uptake_at_pressure(branch, 10**4.5, method="linear", tolerance=0.05)
    assert midpoint["method"] == "interpolated"
    assert midpoint["value"] == pytest.approx(1.5)
    assert [point["observation_id"] for point in midpoint["sources"]] == [1, 2]
    desorption_ignored = uptake_at_pressure(branch, 5e5, method="linear", tolerance=0.05)
    assert desorption_ignored["value"] == pytest.approx(2.0 + math.log10(5))
    assert uptake_at_pressure(branch, 2e6, method="linear", tolerance=0.05) == {
        "status": "out_of_range"
    }
    near = uptake_at_pressure(branch, 1.04e5, method="nearest", tolerance=0.05)
    assert (near["method"], near["value"]) == ("nearest", 2.0)
    far = uptake_at_pressure(branch, 1.2e5, method="nearest", tolerance=0.05)
    assert far["status"] == "no_point_in_tolerance"

###############################################################################
def test_uptake_at_pressure_rows_report_method_and_status() -> None:
    service, _ = _service()

    response = service.uptake_at_pressure(
        UptakeAtPressureRequest(
            filter=AnalyticsFilter(pressure_basis="absolute"), pressure=2.5, pressure_unit="bar"
        )
    )

    by_name = {row.isotherm: row for row in response.rows}
    assert by_name["a-co2-273"].status == "out_of_range"
    assert by_name["a-co2-273"].value is None
    assert by_name["a-co2-298"].method == "interpolated"
    assert [point.pressure for point in by_name["a-co2-298"].sources] == pytest.approx(
        [0.2, 3.0]
    )

###############################################################################
def test_histogram_bins_linear_and_log_scales() -> None:
    linear = histogram([1.0, 2.0, 3.0, 4.0], bins=3, scale="linear")
    assert [bin_.count for bin_ in linear["bins"]] == [1, 1, 2]
    assert linear["stats"].median == pytest.approx(2.5)

    logarithmic = histogram([0.0, 1.0, 10.0, 100.0], bins=2, scale="log")
    assert logarithmic["excluded"] == 1
    assert [bin_.lower for bin_ in logarithmic["bins"]] == pytest.approx([1.0, 10.0])
    assert [bin_.count for bin_ in logarithmic["bins"]] == [1, 2]

###############################################################################
def test_distribution_and_overview_keep_pressure_bases_apart() -> None:
    service, _ = _service()

    with pytest.raises(ValueError, match="pressure basis"):
        service.distribution(DistributionRequest(variable="pressure"))
    relative = service.distribution(
        DistributionRequest(
            variable="pressure", filter=AnalyticsFilter(pressure_basis="relative"), bins=2
        )
    )
    assert relative.unit == "1"
    assert relative.stats.max == pytest.approx(0.5)

    overview = service.overview()
    assert (overview.isotherms, overview.observations, overview.mixture_isotherms) == (4, 11, 1)
    assert overview.isotherms_by_source == {"nist": 2, "uploaded": 2}
    spans = {span.pressure_basis: span for span in overview.pressure_bases}
    assert spans["relative"].pressure_unit == "1"
    assert spans["relative"].pressure_max == pytest.approx(0.5)
    assert spans["absolute"].pressure_max == pytest.approx(3e5)
    assert overview.top_materials[0].name == "Material A"
    assert sum(bin_.count for bin_ in overview.temperature_histogram) == 4

###############################################################################
def test_analytics_routes_map_invalid_queries_to_client_errors(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # TestClient runs requests on another thread, so use a file-backed database.
    service, _ = _service(str(tmp_path / "analytics.db"))
    router = APIRouter(prefix="/analytics")
    AnalyticsEndpoint(router=router, service=service).add_routes()
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    client = TestClient(app)

    assert client.get("/api/v1/analytics/overview").status_code == 200
    assert client.post("/api/v1/analytics/selection", json={}).status_code == 200
    assert client.post("/api/v1/analytics/series", json={}).status_code == 400
    assert (
        client.post(
            "/api/v1/analytics/series",
            json={"filter": {"pressure_basis": "absolute"}, "pressure_unit": "furlong"},
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/api/v1/analytics/distribution",
            json={
                "variable": "temperature_k",
                "filter": {"temperature_min_k": 400, "temperature_max_k": 300},
            },
        ).status_code
        == 422
    )
