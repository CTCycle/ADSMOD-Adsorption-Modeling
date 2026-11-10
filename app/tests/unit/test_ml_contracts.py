# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations
import inspect
import pytest
from server.services.ml_container import MlServiceContainer
from types import SimpleNamespace
from server.models.training.manager import TrainingManager, TrainingProcessRunner
from server.services.data.sequences import PressureUptakeSeriesProcess
import pandas as pd


# Ml boundary


###############################################################################
def test_ml_container_consumes_in_process_snapshot_access() -> None:
    parameters = inspect.signature(MlServiceContainer).parameters
    assert "snapshot_access" in parameters
    assert "internal_token" not in parameters


# Resume validation


###############################################################################
def test_resume_validation_accepts_keras3_public_loss_state() -> None:
    optimizer = SimpleNamespace(variables=[object()])
    model = SimpleNamespace(optimizer=optimizer, loss=object())
    TrainingProcessRunner.validate_resume_model(object(), model)


###############################################################################
def test_resume_validation_rejects_missing_loss() -> None:
    optimizer = SimpleNamespace(variables=[object()])
    model = SimpleNamespace(optimizer=optimizer, loss=None)
    with pytest.raises(ValueError, match="not compiled"):
        TrainingProcessRunner.validate_resume_model(object(), model)


###############################################################################
def test_reconstructed_history_uses_one_based_epochs() -> None:
    manager = object.__new__(TrainingManager)

    history = manager.build_history_entries(
        {
            "history": {
                "loss": [0.5],
                "val_loss": [0.6],
                "MaskedR2": [0.25],
                "val_MaskedR2": [0.3],
            }
        }
    )

    assert history == [
        {
            "epoch": 1,
            "loss": 0.5,
            "val_loss": 0.6,
            "masked_r2": 0.25,
            "val_masked_r2": 0.3,
        }
    ]

###############################################################################
def test_training_snapshot_screening_excludes_multi_species_and_non_pascal_pressure() -> None:
    from server.services.training_data import TrainingDataService

    def row(isotherm_id: int, adsorbate_id: int, unit: str) -> dict:
        return {"isotherm_id": isotherm_id, "adsorbate_id": adsorbate_id, "pressure_canonical_unit": unit}

    rows = [row(1, 10, "Pa"), row(1, 10, "Pa"), row(2, 20, "Pa"), row(2, 21, "Pa"), row(3, 30, "1")]

    kept, dropped = TrainingDataService._screen_training_rows(rows)

    assert [item["isotherm_id"] for item in kept] == [1, 1]
    assert dropped == {"multi_species_isotherms": 1, "non_pascal_pressure_isotherms": 1}
    assert all("adsorbate_id" not in item for item in kept)

###############################################################################
def test_leading_zero_trim_keeps_pressure_uptake_pairs_aligned() -> None:
    frame = pd.DataFrame(
        {"pressure": [[0.0, 0.0, 1.0, 2.0]], "adsorbed_amount": [[0.0, 1.0, 2.0, 3.0]]}
    )

    trimmed = PressureUptakeSeriesProcess({}).remove_leading_zeros(frame)

    assert trimmed.loc[0, "pressure"] == [0.0, 0.0, 1.0, 2.0]
    assert trimmed.loc[0, "adsorbed_amount"] == [0.0, 1.0, 2.0, 3.0]

###############################################################################
def test_long_series_are_subsampled_over_full_range_in_lockstep() -> None:
    processor = PressureUptakeSeriesProcess({"max_measurements": 5})
    pressure = [float(i) for i in range(20)]
    frame = pd.DataFrame({"pressure": [pressure], "adsorbed_amount": [pressure[::-1]]})

    filtered = processor.filter_by_sequence_size(frame)

    kept_pressure = filtered.loc[0, "pressure"]
    kept_uptake = filtered.loc[0, "adsorbed_amount"]
    assert len(kept_pressure) == len(kept_uptake) == 5
    assert kept_pressure[0] == 0.0 and kept_pressure[-1] == 19.0
    assert [p + q for p, q in zip(kept_pressure, kept_uptake)] == [19.0] * 5
