from __future__ import annotations

from typing import Any

import pandas as pd

from adsmod_ml.clients.core_client import CoreSnapshotClient


class DatasetCompositionService:
    """Resolve training sources through Core's immutable snapshot API."""

    required_columns = (
        "filename",
        "temperature",
        "pressure",
        "adsorbed_amount",
        "adsorbate_name",
        "adsorbent_name",
        "pressure_units",
        "adsorption_units",
    )

    def __init__(self, snapshot_client: CoreSnapshotClient) -> None:
        self.snapshot_client = snapshot_client

    def list_sources(self) -> list[dict[str, Any]]:
        sources = self.snapshot_client.list_sources()
        return sorted(
            sources,
            key=lambda item: (
                str(item.get("source", "")) != "nist",
                str(item.get("display_name", "")).casefold(),
            ),
        )

    def compose_datasets(
        self,
        selections: list[dict[str, Any]],
    ) -> tuple[pd.DataFrame, None, None, str]:
        if not selections:
            raise ValueError("No datasets were selected for processing.")
        reference = self.snapshot_client.create_snapshot_from_selections(
            selections,
            metadata={"purpose": "ml_training_source"},
        )
        snapshot = self.snapshot_client.fetch_snapshot(reference.snapshot_id)
        frame = pd.DataFrame(list(snapshot.rows))
        self._require_columns(frame)
        frame = self._normalize_frame(frame)
        if frame.empty:
            raise ValueError("No adsorption data was available after composition.")
        labels = [str(item.get("dataset_name", "")).strip() for item in selections]
        dataset_label = "+".join(label for label in labels if label)[:120]
        return frame, None, None, dataset_label or "composed"

    def _require_columns(self, frame: pd.DataFrame) -> None:
        missing = [column for column in self.required_columns if column not in frame.columns]
        if missing:
            raise ValueError(
                "Core snapshot is missing required training columns: "
                + ", ".join(missing)
            )

    @staticmethod
    def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
        normalized = frame.copy()
        for column in ("temperature", "pressure", "adsorbed_amount"):
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        normalized = normalized.dropna(
            subset=["temperature", "pressure", "adsorbed_amount", "adsorbate_name", "adsorbent_name"]
        )
        normalized = normalized[normalized["temperature"] > 0]
        normalized = normalized[normalized["pressure"] >= 0]
        normalized = normalized[normalized["adsorbed_amount"] >= 0]
        for column in ("adsorbate_name", "adsorbent_name"):
            normalized[column] = normalized[column].astype("string").str.strip().str.lower()
        if "adsorbate_SMILE" not in normalized.columns:
            normalized["adsorbate_SMILE"] = pd.NA
        if "adsorbate_molecular_weight" not in normalized.columns:
            normalized["adsorbate_molecular_weight"] = pd.NA
        return normalized.reset_index(drop=True)


__all__ = ["DatasetCompositionService"]
