import pandas as pd
import pytest
from pydantic import ValidationError

from ml_service.domain.training import TrainingMetadata
from ml_service.learning.serialization.model import ModelSerializer
from ml_service.learning.serialization.training import TrainingDataSerializer

###############################################################################
class StubTrainingQueries:

    # -------------------------------------------------------------------------
    def __init__(self, captured: dict[str, pd.DataFrame]) -> None:
        self.captured = captured

    # -------------------------------------------------------------------------
    def load_training_dataset(self, limit=None):  # noqa: ANN001
        return pd.DataFrame()

    # -------------------------------------------------------------------------
    def upsert_training_dataset(self, dataset):  # noqa: ANN001
        self.captured["upsert"] = dataset.copy()

    # -------------------------------------------------------------------------
    def save_training_metadata(self, metadata):  # noqa: ANN001
        self.captured["metadata"] = metadata.copy()

    # -------------------------------------------------------------------------
    def load_training_metadata(self):
        return pd.DataFrame()


# Helper to create a basis metadata object

###############################################################################
def create_basis_metadata(**kwargs):
    defaults = {
        "sample_size": 1.0,
        "validation_size": 0.2,
        "min_measurements": 1,
        "max_measurements": 30,
        "smile_sequence_size": 20,
        "max_pressure": 10000.0,
        "max_uptake": 20.0,
        "smile_vocabulary": {"C": 1, "H": 2, "O": 3},
        "adsorbent_vocabulary": {"MOF-1": 1, "ZEOLITE-X": 2},
        "normalization_stats": {"pressure_mean": 5000.0, "pressure_std": 2000.0},
    }
    defaults.update(kwargs)
    return TrainingMetadata(**defaults)

###############################################################################
def test_validate_metadata_identical():
    """Verify that two identical metadata objects pass validation."""
    meta1 = create_basis_metadata()
    meta2 = create_basis_metadata()
    assert TrainingDataSerializer.validate_metadata(meta1, meta2) is True

###############################################################################
def test_validate_metadata_param_mismatch():
    """Verify that a scalar parameter mismatch causes validation failure."""
    meta1 = create_basis_metadata(sample_size=1.0)
    meta2 = create_basis_metadata(sample_size=0.5)
    assert TrainingDataSerializer.validate_metadata(meta1, meta2) is False

###############################################################################
def test_validate_metadata_vocab_key_mismatch():
    """Verify that different vocabulary keys cause failure."""
    meta1 = create_basis_metadata(smile_vocabulary={"A": 1})
    meta2 = create_basis_metadata(smile_vocabulary={"A": 1, "B": 2})
    assert TrainingDataSerializer.validate_metadata(meta1, meta2) is False

###############################################################################
def test_validate_metadata_vocab_index_mismatch():
    """Verify that different vocabulary INDICES for same keys cause failure (strict check)."""
    meta1 = create_basis_metadata(smile_vocabulary={"A": 1, "B": 2})
    meta2 = create_basis_metadata(smile_vocabulary={"A": 2, "B": 1})
    assert TrainingDataSerializer.validate_metadata(meta1, meta2) is False

###############################################################################
def test_validate_metadata_normalization_stats():
    """Verify that normalization stats differences cause failure."""
    meta1 = create_basis_metadata(normalization_stats={"mean": 0.0})
    meta2 = create_basis_metadata(normalization_stats={"mean": 0.1})
    assert TrainingDataSerializer.validate_metadata(meta1, meta2) is False

###############################################################################
def test_compute_metadata_hash_determinism():
    """Verify that hash computation is deterministic (order independent for dicts)."""
    meta1 = create_basis_metadata(smile_vocabulary={"A": 1, "B": 2})
    # Create with different insertion order if possible (standard dicts allow this)
    # But params passed to TrainingMetadata are kwargs, so we construct slightly differently
    # to test resilience, but mainly relying on implementation sorting keys.
    meta2 = create_basis_metadata(smile_vocabulary={"B": 2, "A": 1})

    hash1 = TrainingDataSerializer.compute_metadata_hash(meta1)
    hash2 = TrainingDataSerializer.compute_metadata_hash(meta2)
    assert hash1 == hash2

###############################################################################
def test_save_training_dataset_deduplicates_sample_keys():
    captured: dict[str, pd.DataFrame] = {}
    serializer = TrainingDataSerializer(queries=StubTrainingQueries(captured))
    dataset = pd.DataFrame(
        [
            {
                "dataset_name": "NIST ISODB",
                "split": "train",
                "temperature": 298.15,
                "pressure": "[1.0,2.0]",
                "adsorbed_amount": "[0.1,0.2]",
                "encoded_adsorbent": 1,
                "adsorbate_molecular_weight": 44.0,
                "adsorbate_encoded_SMILE": "[1,2,3]",
            },
            {
                "dataset_name": "NIST ISODB",
                "split": "train",
                "temperature": 298.15,
                "pressure": "[1.0,2.0]",
                "adsorbed_amount": "[0.1,0.2]",
                "encoded_adsorbent": 1,
                "adsorbate_molecular_weight": 44.0,
                "adsorbate_encoded_SMILE": "[1,2,3]",
            },
        ]
    )

    serializer.save_training_dataset(
        dataset,
        dataset_label="small_dataset",
        dataset_hash="a" * 64,
    )

    upserted = captured["upsert"]
    assert len(upserted) == 1
    assert upserted["sample_key"].nunique() == 1
    assert upserted.iloc[0]["pressure"] == [1.0, 2.0]
    assert upserted.iloc[0]["adsorbed_amount"] == [0.1, 0.2]
    assert upserted.iloc[0]["adsorbate_encoded_SMILE"] == [1, 2, 3]

###############################################################################
def test_save_training_metadata_normalizes_json_mappings():
    captured: dict[str, pd.DataFrame] = {}
    serializer = TrainingDataSerializer(queries=StubTrainingQueries(captured))
    metadata = pd.DataFrame(
        [
            {
                "dataset_label": "small_dataset",
                "dataset_hash": "a" * 64,
                "smile_vocabulary": '{"C": 1}',
                "adsorbent_vocabulary": '{"MOF-1": 0}',
                "normalization_stats": '{"pressure_mean": 1.0}',
            }
        ]
    )

    serializer.save_training_metadata(metadata, dataset_label="small_dataset")

    saved = captured["metadata"].iloc[0]
    assert saved["smile_vocabulary"] == {"C": 1}
    assert saved["adsorbent_vocabulary"] == {"MOF-1": 0}
    assert saved["normalization_stats"] == {"pressure_mean": 1.0}

###############################################################################
def test_training_dataset_persistence_requires_canonical_hash():
    serializer = TrainingDataSerializer(queries=StubTrainingQueries({}))

    with pytest.raises(ValueError, match="dataset_hash"):
        serializer.save_training_dataset(
            pd.DataFrame({"split": ["train"]}),
            dataset_label="small_dataset",
        )

###############################################################################
def test_training_metadata_rejects_legacy_fields():
    with pytest.raises(ValidationError):
        TrainingMetadata(hashcode="a" * 64)

###############################################################################
def test_checkpoint_metadata_rejects_legacy_hash_alias(tmp_path):
    configuration_dir = tmp_path / "configuration"
    configuration_dir.mkdir()
    (configuration_dir / "configuration.json").write_text("{}", encoding="utf-8")
    (configuration_dir / "metadata.json").write_text(
        '{"hash_code": "' + ("a" * 64) + '"}', encoding="utf-8"
    )
    (configuration_dir / "session_history.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValidationError):
        ModelSerializer().load_training_configuration(str(tmp_path))
