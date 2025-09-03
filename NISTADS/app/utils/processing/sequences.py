from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences

from NISTADS.app.constants import PAD_VALUE
from NISTADS.app.logger import logger


# [MERGE DATASETS]
###############################################################################
class PressureUptakeSeriesProcess:
    def __init__(self, configuration: dict[str, Any]) -> None:
        self.P_COL = "pressure"
        self.Q_COL = "adsorbed_amount"
        self.max_points = configuration.get("max_measurements", 30)
        self.min_points = configuration.get("min_measurements", 1)
        self.max_pressure = configuration.get("max_pressure", 10000) * 1000
        self.max_uptake = configuration.get("max_uptake", 10)

    # -------------------------------------------------------------------------
    def trim_series(self, series: list | np.ndarray) -> list | np.ndarray:
        arr = np.asarray(series)
        nonzero_indices = np.flatnonzero(arr)
        start_idx = max(nonzero_indices[0] - 1, 0) if nonzero_indices.size > 0 else 0

        return series[start_idx:]

    # -------------------------------------------------------------------------
    def remove_leading_zeros(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        # remove contiguous leading zeros from pressure and uptake series
        # leaves a single zero value at the series start if present
        dataframe[self.P_COL] = [self.trim_series(p) for p in dataframe[self.P_COL]]
        dataframe[self.Q_COL] = [self.trim_series(q) for q in dataframe[self.Q_COL]]

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
        # Remove rows where sequence length < min_points
        filtered_data = dataset[
            dataset[self.P_COL].apply(lambda x: len(x) >= self.min_points)
        ].copy()
        # Truncate both columns
        for col in [self.P_COL, self.Q_COL]:
            filtered_data[col] = filtered_data[col].apply(
                lambda x: x[: self.max_points] if len(x) > self.max_points else x
            )

        return filtered_data


# [TOKENIZERS]
###############################################################################
class SMILETokenization:
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
        self.SMILE_padding = configuration.get("SMILE_sequence_size", 20)

    # -------------------------------------------------------------------------
    def tokenize_SMILE_string(self, SMILE: str | Any) -> list[Any]:
        tokens = []
        if isinstance(SMILE, str):
            tokens = []
            i = 0
            length = len(SMILE)
            while i < length:
                c = SMILE[i]
                # Handle brackets
                if c == "[":
                    # Start of bracketed atom
                    j = i + 1
                    bracket_content = ""
                    while j < length and SMILE[j] != "]":
                        bracket_content += SMILE[j]
                        j += 1
                    if j == length:
                        logger.error(f"Incorrect SMILE sequence: {SMILE}")
                    # Now bracket_content contains the content inside brackets
                    # We need to extract the element symbol from bracket_content
                    # The element symbol is the first one or two letters after optional isotope
                    m = re.match(r"(\d+)?([A-Z][a-z]?)", bracket_content)
                    if not m:
                        raise ValueError(
                            f"Invalid atom in brackets: [{bracket_content}]"
                        )
                    isotope, element = m.groups()
                    if element not in self.element_symbols:
                        raise ValueError(f"Unknown element symbol: {element}")
                    tokens.append("[" + bracket_content + "]")
                    i = j + 1  # Move past the closing ']'
                # Handle ring closures with '%'
                elif c == "%":
                    # Ring closure with numbers greater than 9
                    if i + 2 < length and SMILE[i + 1 : i + 3].isdigit():
                        tokens.append(SMILE[i : i + 3])
                        i += 3
                    else:
                        logger.error(
                            f"Invalid ring closure with '%' in SMILE string {SMILE}"
                        )
                # Handle two-character organic subset elements outside brackets
                elif c == "C" and i + 1 < length and SMILE[i + 1] == "l":
                    tokens.append("Cl")
                    i += 2
                elif c == "B" and i + 1 < length and SMILE[i + 1] == "r":
                    tokens.append("Br")
                    i += 2
                # Handle one-character organic subset elements outside brackets
                elif c in "BCNOPSFHI":
                    tokens.append(c)
                    i += 1
                # Handle aromatic atoms (lowercase letters)
                elif c in "bcnops":
                    tokens.append(c)
                    i += 1
                # Handle digits for ring closures
                elif c.isdigit():
                    tokens.append(c)
                    i += 1
                # Handle bond symbols
                elif c in "-=#:/$\\":
                    tokens.append(c)
                    i += 1
                # Handle branch symbols
                elif c in "()":
                    tokens.append(c)
                    i += 1
                # Handle chirality '@'
                elif c == "@":
                    # Check if the next character is also '@'
                    if i + 1 < length and SMILE[i + 1] == "@":
                        tokens.append("@@")
                        i += 2
                    else:
                        tokens.append("@")
                        i += 1
                # Handle '+' or '-' charges
                elif c == "+" or c == "-":
                    charge = c
                    j = i + 1
                    while j < length and (SMILE[j] == c or SMILE[j].isdigit()):
                        charge += SMILE[j]
                        j += 1
                    tokens.append(charge)
                    i = j
                # Handle wildcard '*'
                elif c == "*":
                    tokens.append(c)
                    i += 1
                else:
                    logger.debug(f"Unrecognized character '{c}' at position {i}")
                    i += 1

        return tokens

    # -------------------------------------------------------------------------
    def encode_SMILE_tokens(
        self, data: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[Any, int]]:
        SMILE_tokens = set(
            token for tokens in data["adsorbate_tokenized_SMILE"] for token in tokens
        )

        # Map each token to a unique integer creating an inverted dictionary
        token_to_id = {token: idx for idx, token in enumerate(sorted(SMILE_tokens))}
        # Encode each tokenized SMILE sequence
        data["adsorbate_encoded_SMILE"] = data["adsorbate_tokenized_SMILE"].apply(
            lambda tokens: [int(token_to_id[token]) for token in tokens]
        )

        return data, token_to_id

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
    def process_SMILE_sequences(
        self, dataset: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[Any, int]]:
        dataset["adsorbate_tokenized_SMILE"] = dataset["adsorbate_SMILE"].apply(
            lambda x: self.tokenize_SMILE_string(x)
        )

        dataset, smile_vocabulary = self.encode_SMILE_tokens(dataset)
        dataset = self.SMILE_series_padding(dataset)

        return dataset, smile_vocabulary
