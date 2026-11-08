# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from server.common.constants import PAD_VALUE
from server.common.units import UnitConversionError
from server.models.training.metrics import MaskedRSquared
from server.services.data.conversion import PQ_units_conversion


###############################################################################
def test_masked_r_squared_is_independent_of_batching() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.normal(5.0, 3.0, size=(7, 4)).astype("float32")
    y_pred = (y_true + rng.normal(0.0, 1.0, size=y_true.shape)).astype("float32")
    y_true[:, -1] = PAD_VALUE
    y_true[2, 1] = PAD_VALUE

    valid = y_true != PAD_VALUE
    sse = np.sum((y_true[valid] - y_pred[valid]) ** 2)
    sst = np.sum((y_true[valid] - y_true[valid].mean()) ** 2)
    expected = 1.0 - sse / sst

    for batch_size in (1, 3, len(y_true)):
        metric = MaskedRSquared()
        for start in range(0, len(y_true), batch_size):
            stop = start + batch_size
            metric.update_state(y_true[start:stop], y_pred[start:stop, :, None])
        assert float(metric.result()) == pytest.approx(expected, abs=1e-4)


###############################################################################
def test_unit_conversion_matches_reference_values() -> None:
    frame = pd.DataFrame(
        {
            "pressure": [1.0, 2.0],
            "pressure_units": ["bar", "Pa"],
            "adsorbed_amount": [4.0, 4.0],
            "adsorption_units": ["mmol/g", "mmol/kg"],
            "adsorbate_molecular_weight": [44.0, 44.0],
        }
    )
    converted = PQ_units_conversion(frame)
    assert converted["pressure"].tolist() == pytest.approx([1e5, 2.0])
    assert converted["adsorbed_amount"].tolist() == pytest.approx([4.0, 0.004])


###############################################################################
@pytest.mark.parametrize(
    "row",
    [
        {"pressure_units": "furlongs", "adsorption_units": "mol/kg"},
        {"pressure_units": "Pa", "adsorption_units": "bushels"},
        {"pressure_units": "Pa", "adsorption_units": "mg/g"},
    ],
)
def test_unit_conversion_fails_closed(row: dict[str, str]) -> None:
    frame = pd.DataFrame(
        {
            "pressure": [1.0],
            "adsorbed_amount": [1.0],
            "adsorbate_molecular_weight": [None],
            **{key: [value] for key, value in row.items()},
        }
    )
    with pytest.raises(UnitConversionError):
        PQ_units_conversion(frame)
