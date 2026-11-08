# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences

from server.common.constants import PAD_VALUE
from server.common.utils.logger import logger

###############################################################################
TWO_CHAR_TOKENS = frozenset({"Cl", "Br", "@@"})
SINGLE_CHAR_TOKENS = frozenset("BCNOPSFHIbcnops-=#:/$\\()@*")


###############################################################################
def read_charge(SMILE: str, start: int) -> tuple[str, int]:
    sign = SMILE[start]
    end = start + 1
    while end < len(SMILE) and (SMILE[end] == sign or SMILE[end].isdigit()):
        end += 1
    return SMILE[start:end], end


###############################################################################
class PressureUptakeSeriesProcess:

    # -------------------------------------------------------------------------
    def __init__(self, configuration: dict[str, Any]) -> None:
        self.P_COL = "pressure"
        self.Q_COL = "adsorbed_amount"
        self.max_points = configuration.get("max_measurements", 30)
        self.min_points = configuration.get("min_measurements", 1)
        self.max_pressure = configuration.get("max_pressure", 10000) * 1000
        self.max_uptake = configuration.get("max_uptake", 10)

    # -------------------------------------------------------------------------
    def trim_pairs(
        self, pressure: list[float], uptake: list[float]
    ) -> tuple[list[float], list[float]]:
        """Drop leading all-zero measurements, keeping one, with one shared offset."""
        if len(pressure) != len(uptake):
            raise ValueError("Pressure and uptake series must have equal length.")
        nonzero = np.flatnonzero((np.asarray(pressure) != 0) | (np.asarray(uptake) != 0))
        start_idx = max(int(nonzero[0]) - 1, 0) if nonzero.size > 0 else 0

        return pressure[start_idx:], uptake[start_idx:]

    # -------------------------------------------------------------------------
    def remove_leading_zeros(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        trimmed = [
            self.trim_pairs(p, q)
            for p, q in zip(dataframe[self.P_COL], dataframe[self.Q_COL], strict=True)
        ]
        dataframe[self.P_COL] = [pair[0] for pair in trimmed]
        dataframe[self.Q_COL] = [pair[1] for pair in trimmed]

        return dataframe

    # -------------------------------------------------------------------------
    def PQ_series_padding(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset[self.P_COL] = pad_sequences(
            dataset[self.P_COL],
            maxlen=self.max_points,
            value=PAD_VALUE,
            dtype="float32",
            padding="post",
        ).tolist()

        dataset[self.Q_COL] = pad_sequences(
            dataset[self.Q_COL],
            maxlen=self.max_points,
            value=PAD_VALUE,
            dtype="float32",
            padding="post",
        ).tolist()

        return dataset

    # -------------------------------------------------------------------------
    def filter_by_sequence_size(self, dataset: pd.DataFrame) -> pd.DataFrame:
        filtered_data = dataset[
            dataset[self.P_COL].apply(lambda x: len(x) >= self.min_points)
        ].copy()
        for col in [self.P_COL, self.Q_COL]:
            filtered_data[col] = [
                self.subsample_evenly(values) for values in filtered_data[col]
            ]

        return filtered_data

    # -------------------------------------------------------------------------
    def subsample_evenly(self, values: list[float]) -> list[float]:
        """Keep at most ``max_points`` values spread over the full isotherm range."""
        if len(values) <= self.max_points:
            return values
        indices = np.linspace(0, len(values) - 1, self.max_points).round().astype(int)

        return [values[i] for i in indices]

###############################################################################
class SMILETokenization:

    # -------------------------------------------------------------------------
    def __init__(self, configuration: dict[str, Any]) -> None:
        self.element_symbols = [
            "H",
            "He",
            "Li",
            "Be",
            "B",
            "C",
            "N",
            "O",
            "F",
            "Ne",
            "Na",
            "Mg",
            "Al",
            "Si",
            "P",
            "S",
            "Cl",
            "Ar",
            "K",
            "Ca",
            "Sc",
            "Ti",
            "V",
            "Cr",
            "Mn",
            "Fe",
            "Co",
            "Ni",
            "Cu",
            "Zn",
            "Ga",
            "Ge",
            "As",
            "Se",
            "Br",
            "Kr",
            "Rb",
            "Sr",
            "Y",
            "Zr",
            "Nb",
            "Mo",
            "Tc",
            "Ru",
            "Rh",
            "Pd",
            "Ag",
            "Cd",
            "In",
            "Sn",
            "Sb",
            "Te",
            "I",
            "Xe",
            "Cs",
            "Ba",
            "La",
            "Ce",
            "Pr",
            "Nd",
            "Pm",
            "Sm",
            "Eu",
            "Gd",
            "Tb",
            "Dy",
            "Ho",
            "Er",
            "Tm",
            "Yb",
            "Lu",
            "Hf",
            "Ta",
            "W",
            "Re",
            "Os",
            "Ir",
            "Pt",
            "Au",
            "Hg",
            "Tl",
            "Pb",
            "Bi",
            "Po",
            "At",
            "Rn",
            "Fr",
            "Ra",
            "Ac",
            "Th",
            "Pa",
            "U",
            "Np",
            "Pu",
            "Am",
            "Cm",
            "Bk",
            "Cf",
            "Es",
            "Fm",
            "Md",
            "No",
            "Lr",
            "Rf",
            "Db",
            "Sg",
            "Bh",
            "Hs",
            "Mt",
            "Ds",
            "Rg",
            "Cn",
            "Fl",
            "Lv",
            "Ts",
            "Og",
        ]

        self.organic_subset = ["B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I"]
        self.SMILE_padding = configuration.get("smile_sequence_size", 20)

    # -------------------------------------------------------------------------
    def tokenize_smile_string(self, SMILE: str | Any) -> list[Any]:
        tokens: list[Any] = []
        if not isinstance(SMILE, str):
            return tokens
        i = 0
        length = len(SMILE)
        while i < length:
            c = SMILE[i]
            pair = SMILE[i : i + 2]
            if c == "[":
                token, i = self._read_bracket_atom(SMILE, i)
                tokens.append(token)
            elif c == "%":
                if i + 2 < length and SMILE[i + 1 : i + 3].isdigit():
                    tokens.append(SMILE[i : i + 3])
                    i += 3
                else:
                    logger.error(f"Invalid ring closure with '%' in SMILE string {SMILE}")
                    i += 1
            elif pair in TWO_CHAR_TOKENS:
                tokens.append(pair)
                i += 2
            elif c in SINGLE_CHAR_TOKENS or c.isdigit():
                tokens.append(c)
                i += 1
            elif c == "+":
                token, i = read_charge(SMILE, i)
                tokens.append(token)
            else:
                logger.debug(f"Unrecognized character '{c}' at position {i}")
                i += 1
        return tokens

    # -------------------------------------------------------------------------
    def _read_bracket_atom(self, SMILE: str, start: int) -> tuple[str, int]:
        end = SMILE.find("]", start + 1)
        if end == -1:
            logger.error(f"Incorrect SMILE sequence: {SMILE}")
            end = len(SMILE)
        bracket_content = SMILE[start + 1 : end]
        match = re.match(r"(\d+)?([A-Z][a-z]?)", bracket_content)
        if not match:
            raise ValueError(f"Invalid atom in brackets: [{bracket_content}]")
        element = match.group(2)
        if element not in self.element_symbols:
            raise ValueError(f"Unknown element symbol: {element}")
        return "[" + bracket_content + "]", end + 1

    # -------------------------------------------------------------------------
    def encode_smile_tokens(
        self, data: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[Any, int]]:
        SMILE_tokens = set(
            token for tokens in data["adsorbate_tokenized_SMILE"] for token in tokens
        )

        token_to_id = {
            token: idx for idx, token in enumerate(sorted(SMILE_tokens), start=1)
        }
        data["adsorbate_encoded_SMILE"] = data["adsorbate_tokenized_SMILE"].apply(
            lambda tokens: [int(token_to_id[token]) for token in tokens]
        )

        return data, token_to_id

    # -------------------------------------------------------------------------
    def encode_tokens_from_vocabulary(
        self, tokens: list[Any], vocabulary: dict[Any, int]
    ) -> list[int]:
        return [int(vocabulary[token]) for token in tokens]

    # -------------------------------------------------------------------------
    def encode_smile_tokens_from_vocab(
        self, data: pd.DataFrame, vocabulary: dict[Any, int]
    ) -> tuple[pd.DataFrame, dict[Any, int]]:
        if not vocabulary:
            return pd.DataFrame(), {}

        data["adsorbate_encoded_SMILE"] = [
            self.encode_tokens_from_vocabulary(tokens, vocabulary)
            for tokens in data["adsorbate_tokenized_SMILE"]
        ]

        return data, vocabulary

    # -------------------------------------------------------------------------
    def SMILE_series_padding(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset["adsorbate_encoded_SMILE"] = pad_sequences(
            dataset["adsorbate_encoded_SMILE"],
            maxlen=self.SMILE_padding,
            value=PAD_VALUE,
            dtype="float32",
            padding="post",
        ).tolist()

        return dataset

    # -------------------------------------------------------------------------
    def process_smile_sequences(
        self, dataset: pd.DataFrame, reference_vocabulary: dict[Any, int] | None = None
    ) -> tuple[pd.DataFrame, dict[Any, int]]:
        dataset["adsorbate_tokenized_SMILE"] = dataset["adsorbate_SMILE"].apply(
            lambda x: self.tokenize_smile_string(x)
        )
        token_sizes = dataset["adsorbate_tokenized_SMILE"].apply(len)
        valid_mask = token_sizes > 0
        if not valid_mask.all():
            dropped = int((~valid_mask).sum())
            logger.warning("Dropped %d rows with empty SMILE tokenization", dropped)
        dataset = dataset.loc[valid_mask].copy()
        if dataset.empty:
            return dataset, {}

        if reference_vocabulary is not None:
            missing_tokens = {
                token
                for tokens in dataset["adsorbate_tokenized_SMILE"]
                for token in tokens
                if token not in reference_vocabulary
            }
            if missing_tokens:
                logger.warning(
                    "SMILE vocabulary mismatch; %d tokens not in reference vocabulary",
                    len(missing_tokens),
                )
                return pd.DataFrame(), {}

            dataset, smile_vocabulary = self.encode_smile_tokens_from_vocab(
                dataset, reference_vocabulary
            )
        else:
            dataset, smile_vocabulary = self.encode_smile_tokens(dataset)
        dataset = self.SMILE_series_padding(dataset)

        return dataset, smile_vocabulary
