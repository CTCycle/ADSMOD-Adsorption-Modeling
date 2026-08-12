from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd
from sqlalchemy import delete, select

from shared.common.settings import get_server_settings
from shared.repositories.database.manager import DatabaseManager
from shared.repositories.schemas.models import TrainingDataset, TrainingSample

###############################################################################
class TrainingRepositoryQueries:
    """DataFrame boundary backed by the canonical typed training repository."""

    # -------------------------------------------------------------------------
    def __init__(self, database: DatabaseManager | None = None) -> None:
        self.database = database or DatabaseManager(get_server_settings().database)

    # -------------------------------------------------------------------------
    @staticmethod
    def _sample_frame(rows: list[TrainingSample], label: str, content_hash: str) -> pd.DataFrame:
        values = [
            {
                "dataset_label": label,
                "dataset_hash": content_hash,
                "split": row.split,
                "temperature": row.temperature_k,
                "pressure": row.pressure_values,
                "adsorbed_amount": row.uptake_values,
                "encoded_adsorbent": row.encoded_adsorbent,
                "adsorbate_molecular_weight": row.adsorbate_molar_mass,
                "adsorbate_encoded_SMILE": row.encoded_smiles,
                "sample_key": row.sample_key,
            }
            for row in rows
        ]
        return pd.DataFrame(values)

    # -------------------------------------------------------------------------
    def _get_or_create_parent(self, session: Any, label: str, content_hash: str) -> TrainingDataset:
        parent = session.scalar(
            select(TrainingDataset)
            .where(
                TrainingDataset.label == label,
                TrainingDataset.content_hash == content_hash,
            )
            .order_by(TrainingDataset.id)
        )
        if parent is None:
            parent = session.scalar(
                select(TrainingDataset)
                .where(TrainingDataset.label == label)
                .order_by(TrainingDataset.id)
            )
        if parent is None:
            parent = session.scalar(
                select(TrainingDataset).where(
                    TrainingDataset.content_hash == content_hash
                )
            )
        if parent is None:
            parent = TrainingDataset(content_hash=content_hash, label=label, sample_fraction=1.0, validation_fraction=0.0)
            session.add(parent)
            session.flush()
        else:
            parent.label = label
            parent.content_hash = content_hash
        return parent

    # -------------------------------------------------------------------------
    def load_training_dataset(self, limit: int | None = None) -> pd.DataFrame:
        statement = select(TrainingDataset, TrainingSample).join(TrainingSample, TrainingSample.training_dataset_id == TrainingDataset.id).order_by(TrainingSample.id)
        if limit is not None:
            statement = statement.limit(max(0, int(limit)))
        with self.database.session_factory() as session:
            rows = session.execute(statement).all()
            frames = [self._sample_frame([sample], parent.label, parent.content_hash) for parent, sample in rows]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # -------------------------------------------------------------------------
    def upsert_training_dataset(self, dataset: pd.DataFrame) -> None:
        if dataset.empty:
            return
        required_columns = {"dataset_label", "dataset_hash"}
        missing_columns = required_columns.difference(dataset.columns)
        if missing_columns:
            raise ValueError(
                "Training dataset is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )
        with self.database.transaction() as session:
            for label, group in dataset.groupby("dataset_label", dropna=False):
                normalized_label = str(label).strip()
                if not normalized_label:
                    raise ValueError("Training dataset label must not be empty.")
                records = group.to_dict(orient="records")
                hashes = {
                    str(row["dataset_hash"]).strip()
                    for row in records
                    if isinstance(row.get("dataset_hash"), str)
                    and row["dataset_hash"].strip()
                }
                if len(hashes) != 1:
                    raise ValueError(
                        f"Training dataset '{normalized_label}' must contain one non-empty dataset_hash."
                    )
                content_hash = hashes.pop()
                parent = self._get_or_create_parent(session, normalized_label, content_hash)
                session.execute(delete(TrainingSample).where(TrainingSample.training_dataset_id == parent.id))
                sample_records_by_key = {
                    record["sample_key"]: record
                    for record in (self._sample_record(row, parent.id) for row in records)
                }
                sample_records = list(sample_records_by_key.values())
                session.add_all(TrainingSample(**record) for record in sample_records)
                parent.total_samples = len(sample_records)
                parent.train_samples = sum(record["split"] == "train" for record in sample_records)
                parent.validation_samples = sum(record["split"] == "validation" for record in sample_records)
                parent.test_samples = sum(record["split"] == "test" for record in sample_records)

    # -------------------------------------------------------------------------
    @staticmethod
    def _sample_record(row: dict[str, Any], parent_id: int) -> dict[str, Any]:
        payload = {key: row.get(key) for key in ("split", "temperature", "pressure", "adsorbed_amount", "encoded_adsorbent", "adsorbate_molecular_weight", "adsorbate_encoded_smile")}
        sample_key = row.get("sample_key") or hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
        return {
            "training_dataset_id": parent_id,
            "sample_key": str(sample_key),
            "split": str(row.get("split") or "train"),
            "temperature_k": float(row.get("temperature") or 0),
            "pressure_values": row.get("pressure") or [],
            "uptake_values": row.get("adsorbed_amount") or [],
            "encoded_adsorbent": row.get("encoded_adsorbent"),
            "adsorbate_molar_mass": row.get("adsorbate_molecular_weight"),
            "encoded_smiles": row.get("adsorbate_encoded_SMILE"),
        }

    # -------------------------------------------------------------------------
    def load_training_metadata(self) -> pd.DataFrame:
        with self.database.session_factory() as session:
            rows = session.scalars(select(TrainingDataset).order_by(TrainingDataset.id)).all()
        return pd.DataFrame([
            {
                "dataset_label": row.label,
                "dataset_hash": row.content_hash,
                "created_at": row.created_at,
                "min_measurements": row.min_measurements or 1,
                "max_measurements": row.max_measurements or 30,
                "smile_sequence_size": row.configuration.get("smile_sequence_size", 20),
                "max_pressure": row.configuration.get("max_pressure", 10000.0),
                "max_uptake": row.configuration.get("max_uptake", 20.0),
                "total_samples": row.total_samples,
                "train_samples": row.train_samples,
                "validation_samples": row.validation_samples,
                "test_samples": row.test_samples,
                "sample_size": row.sample_fraction,
                "validation_size": row.validation_fraction,
                "normalization_stats": row.normalization_stats,
                "smile_vocabulary": row.vocabularies.get("smile", []),
                "adsorbent_vocabulary": row.vocabularies.get("adsorbent", []),
            }
            for row in rows
        ])

    # -------------------------------------------------------------------------
    def save_training_metadata(self, metadata: pd.DataFrame) -> None:
        if metadata.empty:
            return
        with self.database.transaction() as session:
            for row in metadata.to_dict(orient="records"):
                label = str(row.get("dataset_label") or "").strip()
                content_hash = str(row.get("dataset_hash") or "").strip()
                if not label or not content_hash:
                    raise ValueError(
                        "Training metadata requires dataset_label and dataset_hash."
                    )
                parent = self._get_or_create_parent(session, label, content_hash)
                parent.sample_fraction = float(row.get("sample_size") or 1.0)
                parent.validation_fraction = float(row.get("validation_size") or 0.0)
                parent.min_measurements = int(row.get("min_measurements") or 1)
                parent.max_measurements = int(row.get("max_measurements") or 30)
                parent.configuration = {
                    "smile_sequence_size": int(row.get("smile_sequence_size") or 20),
                    "max_pressure": float(row.get("max_pressure") or 10000.0),
                    "max_uptake": float(row.get("max_uptake") or 20.0),
                }
                parent.total_samples = int(row.get("total_samples") or 0)
                parent.train_samples = int(row.get("train_samples") or 0)
                parent.validation_samples = int(row.get("validation_samples") or 0)
                parent.test_samples = int(row.get("test_samples") or 0)
                parent.normalization_stats = row.get("normalization_stats") or {}
                parent.vocabularies = {"smile": row.get("smile_vocabulary") or [], "adsorbent": row.get("adsorbent_vocabulary") or []}
