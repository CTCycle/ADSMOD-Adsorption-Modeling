from __future__ import annotations

import pandas as pd
import pytest

from ADSMOD.server.common.constants import DEFAULT_DATASET_COLUMN_MAPPING
from ADSMOD.server.services.modeling.nist_dataset import FittingNISTDatasetService


def test_prepare_nist_dataframe_requires_expected_columns() -> None:
    service = FittingNISTDatasetService()
    with pytest.raises(ValueError, match="missing required columns"):
        service.prepare_nist_dataframe(pd.DataFrame({"name": ["exp_1"]}), pd.DataFrame())


def test_load_for_fitting_preserves_response_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    service = FittingNISTDatasetService()

    nist_rows = pd.DataFrame(
        {
            "name": ["exp_1"],
            "adsorbent": ["mof-5"],
            "adsorbate": ["co2"],
            "temperature": [298.15],
            "pressure": [1000.0],
            "adsorbed_amount": [2.0],
            "pressure_units": ["pa"],
            "adsorption_units": ["mol/kg"],
        }
    )
    adsorbates = pd.DataFrame({"name": ["co2"], "molecular_weight": [44.01]})

    monkeypatch.setattr(
        "ADSMOD.server.services.modeling.nist_dataset.NISTDataSerializer.load_adsorption_datasets",
        lambda _self: (nist_rows, adsorbates, pd.DataFrame()),
    )
    monkeypatch.setattr(
        service,
        "prepare_nist_dataframe",
        lambda _nist_df, _adsorbates_df: pd.DataFrame(
            {
                "filename": ["exp_1"],
                "experiment": ["exp_1_mof-5_co2_298.15K"],
                "temperature": [298.15],
                "pressure": [1000.0],
                "adsorbed_amount": [0.002],
            }
        ),
    )

    response = service.load_for_fitting()
    assert response.status == "success"
    assert response.dataset.dataset_name == "nist_single_component"
    assert response.dataset.row_count == 1
    assert set(response.dataset.columns) == {
        "filename",
        "experiment",
        DEFAULT_DATASET_COLUMN_MAPPING["temperature"],
        DEFAULT_DATASET_COLUMN_MAPPING["pressure"],
        DEFAULT_DATASET_COLUMN_MAPPING["uptake"],
    }
