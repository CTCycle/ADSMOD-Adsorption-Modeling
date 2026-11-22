# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("KERAS_BACKEND", "torch")
pad_reference = pytest.importorskip("keras.preprocessing.sequence").pad_sequences

CASES = {
    "shorter": [[1.0, 2.0]],
    "exact": [[1.0, 2.0, 3.0, 4.0]],
    "longer_keeps_last": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
    "empty": [[]],
    "mixed": [[1.0], [], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]],
}


# -----------------------------------------------------------------------------
@pytest.mark.parametrize("sequences", CASES.values(), ids=CASES.keys())
@pytest.mark.parametrize("as_series", [False, True], ids=["list", "series"])
def test_pad_sequences_matches_keras(sequences: list, as_series: bool) -> None:
    from server.services.data.sequences import pad_sequences

    data = pd.Series(sequences) if as_series else sequences
    expected = pad_reference(
        sequences, maxlen=4, value=-1.0, dtype="float32", padding="post"
    )
    actual = pad_sequences(data, maxlen=4, value=-1.0, dtype="float32")
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


# -----------------------------------------------------------------------------
def test_series_padding_call_sites_pad_to_configured_length() -> None:
    from server.common.constants import PAD_VALUE
    from server.services.data.sequences import (
        PressureUptakeSeriesProcess,
        SMILETokenization,
    )

    frame = pd.DataFrame(
        {
            "pressure": [[1.0, 2.0]],
            "adsorbed_amount": [[3.0, 4.0]],
            "adsorbate_encoded_SMILE": [[5.0]],
        }
    )
    padded = PressureUptakeSeriesProcess({"max_measurements": 3}).PQ_series_padding(
        frame
    )
    padded = SMILETokenization({"smile_sequence_size": 2}).SMILE_series_padding(padded)

    assert padded.loc[0, "pressure"] == [1.0, 2.0, PAD_VALUE]
    assert padded.loc[0, "adsorbed_amount"] == [3.0, 4.0, PAD_VALUE]
    assert padded.loc[0, "adsorbate_encoded_SMILE"] == [5.0, PAD_VALUE]
