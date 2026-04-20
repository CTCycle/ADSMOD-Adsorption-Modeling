from __future__ import annotations

import pytest

from ADSMOD.server.domain.fitting import FittingRequest
from ADSMOD.server.services.fitting import FittingService


def build_request() -> FittingRequest:
    return FittingRequest.model_validate(
        {
            "max_iterations": 10,
            "optimization_method": "LSS",
            "parameter_bounds": {
                "Langmuir": {
                    "min": {"k": 1e-6, "qsat": 0.0},
                    "max": {"k": 10.0, "qsat": 100.0},
                    "initial": {"k": 0.2, "qsat": 5.0},
                }
            },
            "dataset": {
                "dataset_name": "unit_dataset",
                "columns": ["pressure", "uptake", "temperature"],
                "records": [
                    {"pressure": 10.0, "uptake": 1.0, "temperature": 298.15},
                    {"pressure": 20.0, "uptake": 1.5, "temperature": 298.15},
                ],
            },
        }
    )


def test_start_fitting_job_returns_job_start_response(monkeypatch: pytest.MonkeyPatch) -> None:
    service = FittingService()
    payload = build_request()

    monkeypatch.setattr("ADSMOD.server.services.fitting.job_manager.is_job_running", lambda _job_type: False)
    monkeypatch.setattr("ADSMOD.server.services.fitting.job_manager.start_job", lambda **_kwargs: "fit12345")

    response = service.start_fitting_job(payload)
    assert response.job_id == "fit12345"
    assert response.job_type == "fitting"
    assert response.status == "running"


def test_cancel_job_returns_modeled_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    service = FittingService()
    monkeypatch.setattr("ADSMOD.server.services.fitting.job_manager.cancel_job", lambda _job_id: True)

    response = service.cancel_job("fit12345")
    assert response.status == "cancelled"
    assert response.job_id == "fit12345"


def test_cancel_job_raises_when_not_cancellable(monkeypatch: pytest.MonkeyPatch) -> None:
    service = FittingService()
    monkeypatch.setattr("ADSMOD.server.services.fitting.job_manager.cancel_job", lambda _job_id: False)

    with pytest.raises(ValueError, match="cannot be cancelled"):
        service.cancel_job("fit12345")
