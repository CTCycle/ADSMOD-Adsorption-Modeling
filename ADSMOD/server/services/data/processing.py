from __future__ import annotations

import re
from difflib import get_close_matches
from typing import Any

import numpy as np
import pandas as pd

from ADSMOD.server.configurations import server_settings
from ADSMOD.server.common.constants import (
    COLUMN_AIC,
    COLUMN_AICC,
    COLUMN_BEST_MODEL,
    COLUMN_DATASET_NAME,
    COLUMN_EXPERIMENT,
    COLUMN_FILENAME,
    COLUMN_MAX_PRESSURE,
    COLUMN_MAX_UPTAKE,
    COLUMN_MEASUREMENT_COUNT,
    COLUMN_MIN_PRESSURE,
    COLUMN_MIN_UPTAKE,
    COLUMN_OPTIMIZATION_METHOD,
    COLUMN_SCORE,
    COLUMN_WORST_MODEL,
    DEFAULT_DATASET_COLUMN_MAPPING,
)
from ADSMOD.server.common.utils.logger import logger
from ADSMOD.server.entities.datasets import DatasetColumns


###############################################################################
class AdsorptionDataProcessor:
    def __init__(self, dataset: pd.DataFrame) -> None:
        self.dataset = dataset.copy()
        self.columns = DatasetColumns()

    # -------------------------------------------------------------------------
    def preprocess(
        self, detect_columns: bool = True
    ) -> tuple[pd.DataFrame, DatasetColumns, str]:
        """Clean the dataset, infer column mapping, and compute statistics.

        Keyword arguments:
        detect_columns -- Toggle automatic column detection based on heuristics.

        Return value:
        Tuple containing the aggregated dataset, resolved column names, and a
        statistics report.
        """
        if self.dataset.empty:
            raise ValueError("Provided dataset is empty")

        if detect_columns:
            # Column detection harmonizes arbitrary headers with the canonical schema
            # used throughout the pipeline before any filtering happens.
            self.identify_columns()

        cleaned = self.drop_invalid_values(self.dataset)
        grouped = self.aggregate_by_experiment(cleaned)
        filtered = self.filter_invalid_experiments(grouped)
        stats = self.build_statistics(cleaned, filtered)

        return filtered, self.columns, stats

    # -------------------------------------------------------------------------
    def identify_columns(self) -> None:
        """Infer dataset column names that correspond to canonical adsorption fields.

        Keyword arguments:
        None.

        Return value:
        None.
        """
        cutoff = server_settings.datasets.column_detection_cutoff
        for attr, pattern in DEFAULT_DATASET_COLUMN_MAPPING.items():
            matched_cols = [
                column
                for column in self.dataset.columns
                if re.search(pattern.split()[0], column, re.IGNORECASE)
            ]
            if matched_cols:
                # Prefer a direct regex match when a close equivalent column exists.
                setattr(self.columns, attr, matched_cols[0])
                continue
            close_matches = get_close_matches(
                pattern,
                list(self.dataset.columns),
                cutoff=cutoff,
            )
            if close_matches:
                # Fallback to fuzzy matching when naming deviates but is still similar.
                setattr(self.columns, attr, close_matches[0])

    # -------------------------------------------------------------------------
    def drop_invalid_values(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Remove rows containing invalid measurements for the detected adsorption columns.

        Keyword arguments:
        dataset -- Dataset that should be filtered using the resolved column mapping.

        Return value:
        DataFrame limited to valid rows with non-negative measurements and
        temperatures above zero.
        """
        cols = self.columns.as_dict()
        valid = dataset.dropna(subset=list(cols.values()))
        valid = valid[valid[cols["temperature"]].astype(float) > 0]
        valid = valid[valid[cols["pressure"]].astype(float) >= 0]
        valid = valid[valid[cols["uptake"]].astype(float) >= 0]
        return valid.reset_index(drop=True)

    # -------------------------------------------------------------------------
    def aggregate_by_experiment(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Group cleaned measurements by experiment and compute aggregate metrics.

        Keyword arguments:
        dataset -- Filtered dataset containing valid measurements.

        Return value:
        DataFrame with one row per experiment including pressure and uptake vectors and
        summary stats.
        """
        cols = self.columns.as_dict()
        aggregate = {
            cols["temperature"]: "first",
            cols["pressure"]: list,
            cols["uptake"]: list,
        }
        for optional_column in (COLUMN_DATASET_NAME, COLUMN_FILENAME):
            if optional_column in dataset.columns:
                aggregate[optional_column] = "first"
        # ``groupby`` collects all measurements for each experiment so downstream
        # fitting can consume full pressure/uptake vectors.
        grouped = (
            dataset.groupby(cols["experiment"], as_index=False)
            .agg(aggregate)
            .rename(columns={cols["experiment"]: COLUMN_EXPERIMENT})
        )
        grouped[COLUMN_MEASUREMENT_COUNT] = grouped[cols["pressure"]].apply(len)
        grouped[COLUMN_MIN_PRESSURE] = grouped[cols["pressure"]].apply(min)
        grouped[COLUMN_MAX_PRESSURE] = grouped[cols["pressure"]].apply(max)
        grouped[COLUMN_MIN_UPTAKE] = grouped[cols["uptake"]].apply(min)
        grouped[COLUMN_MAX_UPTAKE] = grouped[cols["uptake"]].apply(max)
        return grouped

    # -------------------------------------------------------------------------
    def validate_experiment(self, pressure: Any, uptake: Any) -> bool:
        try:
            pressure_array = np.asarray(pressure, dtype=np.float64)
            uptake_array = np.asarray(uptake, dtype=np.float64)
        except Exception:  # noqa: BLE001
            return False

        if pressure_array.ndim != 1 or uptake_array.ndim != 1:
            return False
        if pressure_array.size < 2 or uptake_array.size < 2:
            return False
        if pressure_array.size != uptake_array.size:
            return False
        if not np.all(np.isfinite(pressure_array)):
            return False
        if not np.all(np.isfinite(uptake_array)):
            return False

        return True

    # -------------------------------------------------------------------------
    def validate_experiment_row(self, row: pd.Series) -> bool:
        cols = self.columns.as_dict()
        return self.validate_experiment(row[cols["pressure"]], row[cols["uptake"]])

    # -------------------------------------------------------------------------
    def filter_invalid_experiments(self, dataset: pd.DataFrame) -> pd.DataFrame:
        valid_mask = dataset.apply(self.validate_experiment_row, axis=1)
        removed_count = int((~valid_mask).sum())
        if removed_count:
            logger.info(
                "Skipped %s invalid experiments during preprocessing", removed_count
            )
        return dataset.loc[valid_mask].reset_index(drop=True)

    # -------------------------------------------------------------------------
    def build_statistics(self, cleaned: pd.DataFrame, grouped: pd.DataFrame) -> str:
        """Produce a Markdown report describing dataset sizes and cleansing outcomes.

        Keyword arguments:
        cleaned -- Dataset after removing invalid rows.
        grouped -- Aggregated dataset produced by :meth:`aggregate_by_experiment`.

        Return value:
        Markdown-formatted string summarizing per-column usage and high-level
        metrics.
        """
        total_measurements = cleaned.shape[0]
        total_experiments = grouped.shape[0]
        removed_nan = self.dataset.shape[0] - total_measurements
        avg_measurements = (
            total_measurements / total_experiments if total_experiments else 0
        )

        stats = (
            "#### Dataset Statistics\n\n"
            f"**Experiments column:** {self.columns.experiment}\n"
            f"**Temperature column:** {self.columns.temperature}\n"
            f"**Pressure column:** {self.columns.pressure}\n"
            f"**Uptake column:** {self.columns.uptake}\n\n"
            f"**Number of NaN values removed:** {removed_nan}\n"
            f"**Number of experiments:** {total_experiments}\n"
            f"**Number of measurements:** {total_measurements}\n"
            f"**Average measurements per experiment:** {avg_measurements:.1f}"
        )
        return stats


###############################################################################
class DatasetAdapter:
    # -------------------------------------------------------------------------
    @staticmethod
    def combine_results(
        fitting_results: dict[str, list[dict[str, Any]]],
        dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        """Append model fitting metrics and parameters to the processed dataset.

        Keyword arguments:
        fitting_results -- Mapping of model names to experiment-level fitting
        diagnostics.
        dataset -- Aggregated dataset to be enriched with fitting outputs.

        Return value:
        DataFrame with additional columns per model containing the optimization
        score, method, and parameter estimates.
        """
        if not fitting_results:
            logger.warning("No fitting results were provided")
            return dataset

        result_df = dataset.copy()
        for model_name, entries in fitting_results.items():
            if not entries:
                logger.info("Model %s produced no entries", model_name)
                continue
            params = entries[0].get("arguments", [])
            # Columns for each model store experiment-level metrics aligned by order.
            result_df[f"{model_name} {COLUMN_SCORE}"] = [
                entry.get("score", np.nan) for entry in entries
            ]
            result_df[f"{model_name} {COLUMN_AIC}"] = [
                entry.get("aic", np.nan) for entry in entries
            ]
            result_df[f"{model_name} {COLUMN_AICC}"] = [
                entry.get("aicc", np.nan) for entry in entries
            ]
            result_df[f"{model_name} {COLUMN_OPTIMIZATION_METHOD}"] = [
                entry.get("optimization_method") for entry in entries
            ]
            for index, param in enumerate(params):
                result_df[f"{model_name} {param}"] = [
                    entry.get("optimal_params", [np.nan] * len(params))[index]
                    for entry in entries
                ]
                result_df[f"{model_name} {param} error"] = [
                    entry.get("errors", [np.nan] * len(params))[index]
                    for entry in entries
                ]
        return result_df

    # -------------------------------------------------------------------------
    @staticmethod
    def compute_best_models(dataset: pd.DataFrame, metric: str) -> pd.DataFrame:
        """Determine the best and worst model per experiment using the chosen metric."""
        suffix = DatasetAdapter.normalize_metric(metric)
        candidates = DatasetAdapter.metric_priority(suffix)

        metric_columns: list[str] = []
        selected_suffix = ""
        for candidate in candidates:
            metric_columns = [
                column for column in dataset.columns if column.endswith(candidate)
            ]
            if metric_columns:
                selected_suffix = candidate
                break

        if not metric_columns:
            logger.info(
                "No columns found for ranking metric %s; best model computation skipped",
                metric,
            )
            return dataset

        ranked = dataset.copy()
        ranked_values = ranked[metric_columns].replace({np.inf: np.nan})
        ranked[COLUMN_BEST_MODEL] = ranked_values.idxmin(axis=1).str.replace(
            f" {selected_suffix}", ""
        )
        ranked[COLUMN_WORST_MODEL] = ranked_values.idxmax(axis=1).str.replace(
            f" {selected_suffix}", ""
        )
        return ranked

    # -------------------------------------------------------------------------
    @staticmethod
    def normalize_metric(metric: str) -> str:
        normalized = metric.replace("-", "").replace("_", "").strip().upper()
        if normalized == "AICC":
            return COLUMN_AICC
        if normalized == "AIC":
            return COLUMN_AIC
        if normalized == "SCORE":
            return COLUMN_SCORE
        return COLUMN_AICC

    # -------------------------------------------------------------------------
    @staticmethod
    def metric_priority(primary: str) -> list[str]:
        order = [COLUMN_AICC, COLUMN_AIC, COLUMN_SCORE]
        prioritized = [primary] + [metric for metric in order if metric != primary]
        seen: set[str] = set()
        unique: list[str] = []
        for metric in prioritized:
            if metric not in seen:
                unique.append(metric)
                seen.add(metric)
        return unique
