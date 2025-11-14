from __future__ import annotations

import math
import os
import re
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import Model
from matplotlib.figure import Figure
from sklearn.cluster import AgglomerativeClustering

from NISTADS.app.client.workers import check_thread_status, update_progress_callback
from NISTADS.app.utils.constants import EVALUATION_PATH, PAD_VALUE
from NISTADS.app.utils.logger import logger
from NISTADS.app.utils.services.loader import SCADSDataLoader
from NISTADS.app.utils.repository.serializer import DataSerializer


###############################################################################
class AdsorptionExperimentsClustering:
    def __init__(self, configuration: dict[str, Any]) -> None:
        self.configuration = configuration
        self.serializer = DataSerializer()
        self.seed = configuration.get("seed", 42)
        self.sample_size = float(configuration.get("sample_size", 1.0))
        self.n_clusters = 4
        self.grid_size = 50
        self.pressure_col = self.serializer.P_COL
        self.uptake_col = self.serializer.Q_COL
        self.group_keys = [
            "filename",
            "temperature",
            "adsorbent_name",
            "adsorbate_name",
        ]
        self.grid = np.linspace(0.0, 1.0, self.grid_size)
        self.output_dir = EVALUATION_PATH
        os.makedirs(self.output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    def evaluate(
        self,
        adsorption_data: pd.DataFrame,
        **kwargs: Any,
    ) -> Figure | None:
        experiments = self.prepare_experiments(adsorption_data)
        if not experiments:
            logger.warning("No valid adsorption experiments available for clustering")
            return None

        if len(experiments) < 2:
            logger.warning("At least two experiments are required for clustering")
            return None

        worker = kwargs.get("worker")
        progress_callback = kwargs.get("progress_callback")
        distance_matrix = self.compute_distance_matrix(
            [exp["normalized"] for exp in experiments],
            worker,
            progress_callback,
        )
        cluster_count = min(self.n_clusters, len(experiments))
        if cluster_count < 2:
            labels = np.zeros(len(experiments), dtype=int)
        else:
            clustering = AgglomerativeClustering(
                n_clusters=cluster_count,
                metric="precomputed",
                linkage="average",
            )
            labels = clustering.fit_predict(distance_matrix)

        figure = self.plot_clusters(experiments, labels)
        self.save_plot(figure)

        return figure

    # -------------------------------------------------------------------------
    def prepare_experiments(
        self,
        adsorption_data: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        required_cols = set(self.group_keys + [self.pressure_col, self.uptake_col])
        if not required_cols.issubset(adsorption_data.columns):
            missing = required_cols.difference(adsorption_data.columns)
            logger.warning(
                "Adsorption dataset is missing required columns: %s",
                ", ".join(sorted(missing)),
            )
            return []

        cleaned = adsorption_data.dropna(subset=list(required_cols))
        total_experiments = cleaned["filename"].nunique(dropna=True)
        selected_filenames = self.select_filenames(cleaned)
        if not selected_filenames:
            logger.warning(
                "No experiments selected for DTW clustering (total=%s, fraction=%s)",
                total_experiments,
                self.sample_size,
            )
            return []

        filtered = cleaned[cleaned["filename"].isin(selected_filenames)]

        experiments: list[dict[str, Any]] = []
        for filename, group in filtered.groupby("filename", sort=False):
            ordered = group.sort_values(self.pressure_col)
            pressure = ordered[self.pressure_col].to_numpy(dtype=float)
            uptake = ordered[self.uptake_col].to_numpy(dtype=float)
            mask = np.isfinite(pressure) & np.isfinite(uptake)
            pressure = pressure[mask]
            uptake = uptake[mask]
            if pressure.size < 3 or np.allclose(pressure, pressure[0]):
                continue

            temperature = float(ordered["temperature"].iloc[0])
            adsorbent = str(ordered["adsorbent_name"].iloc[0])
            adsorbate = str(ordered["adsorbate_name"].iloc[0])

            normalized = self.normalize_curve(pressure, uptake)
            experiments.append(
                {
                    "id": self.format_identifier((filename, temperature, adsorbent, adsorbate)),
                    "pressure": pressure,
                    "uptake": uptake,
                    "normalized": normalized,
                }
            )

        logger.info(
            "Prepared %s experiments for DTW clustering (requested fraction=%s, total available=%s)",
            len(experiments),
            self.sample_size,
            total_experiments,
        )

        return experiments

    # -------------------------------------------------------------------------
    def select_filenames(self, data: pd.DataFrame) -> list[str]:
        filenames = data["filename"].dropna().unique().tolist()
        total = len(filenames)
        if total == 0:
            return []

        fraction = self.sample_size
        if fraction <= 0:
            return []

        target = max(int(math.floor(total * fraction)), 1)

        if target >= total:
            return filenames

        rng = np.random.default_rng(self.seed)
        indices = np.sort(rng.choice(total, size=target, replace=False))

        return [filenames[idx] for idx in indices]

    # -------------------------------------------------------------------------
    def normalize_curve(
        self,
        pressure: np.ndarray,
        uptake: np.ndarray,
    ) -> np.ndarray:
        pressure_min = pressure.min()
        pressure_range = pressure.max() - pressure_min
        uptake_min = uptake.min()
        uptake_range = uptake.max() - uptake_min

        if pressure_range == 0:
            pressure_norm = np.zeros_like(pressure, dtype=float)
        else:
            pressure_norm = (pressure - pressure_min) / pressure_range

        if uptake_range == 0:
            uptake_norm = np.zeros_like(uptake, dtype=float)
        else:
            uptake_norm = (uptake - uptake_min) / uptake_range

        order = np.argsort(pressure_norm)
        pressure_norm = pressure_norm[order]
        uptake_norm = uptake_norm[order]
        unique_pressure, unique_idx = np.unique(pressure_norm, return_index=True)
        uptake_unique = uptake_norm[unique_idx]

        if unique_pressure.size < 2:
            values = np.full_like(
                self.grid, uptake_unique[0] if uptake_unique.size else 0.0
            )
            return np.column_stack((self.grid, values))

        interpolated = np.interp(self.grid, unique_pressure, uptake_unique)
        return np.column_stack((self.grid, interpolated))

    # -------------------------------------------------------------------------
    def compute_distance_matrix(
        self,
        curves: list[np.ndarray],
        worker: Any | None,
        progress_callback: Any | None,
    ) -> np.ndarray:
        size = len(curves)
        distances = np.zeros((size, size), dtype=float)
        total = max(size * (size - 1) // 2, 1)
        completed = 0

        for i in range(size):
            for j in range(i + 1, size):
                check_thread_status(worker)
                distance = self.dtw_distance(curves[i], curves[j])
                distances[i, j] = distance
                distances[j, i] = distance
                completed += 1
                update_progress_callback(completed, total, progress_callback)

        return distances

    # -------------------------------------------------------------------------
    def dtw_distance(
        self,
        series_a: np.ndarray,
        series_b: np.ndarray,
    ) -> float:
        len_a, len_b = len(series_a), len(series_b)
        cost = np.full((len_a + 1, len_b + 1), np.inf, dtype=float)
        cost[0, 0] = 0.0

        for i in range(1, len_a + 1):
            for j in range(1, len_b + 1):
                dist = np.linalg.norm(series_a[i - 1] - series_b[j - 1])
                cost[i, j] = dist + min(
                    cost[i - 1, j],
                    cost[i, j - 1],
                    cost[i - 1, j - 1],
                )

        return cost[len_a, len_b] / (len_a + len_b)

    # -------------------------------------------------------------------------
    def plot_clusters(
        self,
        experiments: list[dict[str, Any]],
        labels: np.ndarray,
    ) -> Figure:
        unique_labels = np.unique(labels)
        cluster_count = len(unique_labels)
        fig, axes = plt.subplots(
            cluster_count,
            1,
            figsize=(8, 3 * cluster_count),
            sharex=True,
            sharey=True,
        )

        if cluster_count == 1:
            axes = [axes]

        cmap = plt.get_cmap("tab10", cluster_count)

        for idx, label in enumerate(unique_labels):
            ax = axes[idx]
            cluster_curves = [
                exp
                for exp, curve_label in zip(experiments, labels)
                if curve_label == label
            ]
            color = cmap(idx)
            for exp in cluster_curves:
                curve = exp["normalized"]
                ax.plot(self.grid, curve[:, 1], color=color, alpha=0.2)

            mean_curve, std_curve = self.cluster_profile(cluster_curves)
            ax.plot(
                self.grid,
                mean_curve,
                color=color,
                linewidth=2.5,
                label=f"Cluster {idx + 1}",
            )
            ax.fill_between(
                self.grid,
                mean_curve - std_curve,
                mean_curve + std_curve,
                color=color,
                alpha=0.1,
            )
            ax.set_title(f"Cluster {idx + 1} ({len(cluster_curves)} experiments)")
            ax.set_ylabel("Normalized uptake")
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.legend(loc="upper left")

        axes[-1].set_xlabel("Normalized pressure")
        fig.suptitle("DTW clustering of adsorption isotherms", fontsize=14)
        fig.tight_layout()

        return fig

    # -------------------------------------------------------------------------
    def cluster_profile(
        self,
        experiments: list[dict[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray]:
        if not experiments:
            zeros = np.zeros_like(self.grid)
            return zeros, zeros

        values = np.vstack([exp["normalized"][:, 1] for exp in experiments])
        return values.mean(axis=0), values.std(axis=0)

    # -------------------------------------------------------------------------
    def save_plot(self, fig: Figure) -> None:
        out_path = os.path.join(
            self.output_dir, "adsorption_experiments_clustering.jpeg"
        )
        fig.savefig(out_path, dpi=300, bbox_inches="tight")

    # -------------------------------------------------------------------------
    def format_identifier(self, keys: tuple[Any, ...]) -> str:
        filename, temperature, adsorbent, adsorbate = keys
        return f"{filename} | {adsorbent} | {adsorbate} | {temperature:.1f}K"


class AdsorptionPredictionsQuality:
    def __init__(
        self,
        model: Model,
        configuration: dict[str, Any],
        metadata: dict,
        checkpoint_path: str,
        num_experiments: int = 6,
    ) -> None:
        self.save_images = configuration.get("save_images", True)
        self.model = model
        self.configuration = configuration
        self.metadata = metadata
        self.dataloader = SCADSDataLoader(configuration, metadata)
        self.num_experiments = num_experiments
        self.cols = int(np.ceil(np.sqrt(self.num_experiments)))
        self.rows = int(np.ceil(self.num_experiments / self.cols))
        self.img_resolution = configuration.get("image_resolution", 400)
        self.file_type = "jpeg"

        self.checkpoint = os.path.basename(checkpoint_path)
        self.validation_path = os.path.join(
            EVALUATION_PATH, "validation", self.checkpoint
        )
        os.makedirs(self.validation_path, exist_ok=True)

    # -------------------------------------------------------------------------
    def save_image(self, fig: Figure, name: str) -> None:
        name = re.sub(r"[^0-9A-Za-z_]", "_", name)
        out_path = os.path.join(self.validation_path, name)
        fig.savefig(out_path, bbox_inches="tight", dpi=self.img_resolution)

    # -------------------------------------------------------------------------
    def process_uptake_curves(
        self,
        inputs: pd.DataFrame | dict,
        output: pd.DataFrame | np.ndarray,
        predicted_output: np.ndarray,
    ) -> tuple[list[Any], list[Any], list[Any]]:
        pressures, uptakes, predictions = [], [], []
        for exp in range(self.num_experiments):
            pressure = inputs["pressure_input"][exp, :]
            true_y = output[exp, :]
            predicted_y = np.squeeze(predicted_output[exp, :])
            # calculate unpadded length to properly truncate series
            valid_pressure = pressure[pressure != PAD_VALUE]
            pressure = pressure[: len(valid_pressure)]
            true_y = true_y[: len(valid_pressure)]
            predicted_y = predicted_y[: len(valid_pressure)]

            pressures.append(pressure * self.metadata["normalization"]["pressure"])
            uptakes.append(true_y * self.metadata["normalization"]["adsorbed_amount"])
            predictions.append(
                predicted_y * self.metadata["normalization"]["adsorbed_amount"]
            )

        return pressures, uptakes, predictions

    # -------------------------------------------------------------------------
    def visualize_adsorption_isotherms(
        self, validation_data: pd.DataFrame, **kwargs
    ) -> Figure | None:
        sampled_data = validation_data.sample(n=self.num_experiments, random_state=42)
        sampled_X, sampled_Y = self.dataloader.separate_inputs_and_output(sampled_data)
        if sampled_Y is None:
            logger.warning(
                "Reference uptake values have not been found, data evaluation is skipped"
            )
            return

        predictions = self.model.predict(sampled_X)
        # process training uptake curves
        check_thread_status(kwargs.get("worker", None))
        pressures, uptakes, predictions = self.process_uptake_curves(
            sampled_X, sampled_Y, predictions
        )

        # Create the subplots (flatten axes to simplify iteration later)
        fig, axes = plt.subplots(
            self.rows, self.cols, figsize=(5 * self.cols, 4 * self.rows)
        )
        axes = np.array(axes).flatten()

        for i in range(self.num_experiments):
            axes[i].plot(pressures[i], uptakes[i], label="Adsorbed amount")
            axes[i].plot(pressures[i], predictions[i], label="Predicted adsorption")
            axes[i].set_title(f"Plot {i + 1}")

            check_thread_status(kwargs.get("worker", None))
            update_progress_callback(
                i + 1, self.num_experiments, kwargs.get("progress_callback", None)
            )

        plt.title("Comparison of validation adsorption isotherms", fontsize=16)
        plt.tight_layout()
        self.save_image(fig, "validation_curves_comparison.jpeg")
        plt.close()

        return fig
