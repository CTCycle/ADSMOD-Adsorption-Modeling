import pytest
from ADSMOD.server.repository.serializer import TrainingDataSerializer
from ADSMOD.server.schemas.training import TrainingMetadata


# Helper to create a basis metadata object
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


def test_validate_metadata_identical():
    """Verify that two identical metadata objects pass validation."""
    meta1 = create_basis_metadata()
    meta2 = create_basis_metadata()
    assert TrainingDataSerializer.validate_metadata(meta1, meta2) is True


def test_validate_metadata_param_mismatch():
    """Verify that a scalar parameter mismatch causes validation failure."""
    meta1 = create_basis_metadata(sample_size=1.0)
    meta2 = create_basis_metadata(sample_size=0.5)
    assert TrainingDataSerializer.validate_metadata(meta1, meta2) is False


def test_validate_metadata_vocab_key_mismatch():
    """Verify that different vocabulary keys cause failure."""
    meta1 = create_basis_metadata(smile_vocabulary={"A": 1})
    meta2 = create_basis_metadata(smile_vocabulary={"A": 1, "B": 2})
    assert TrainingDataSerializer.validate_metadata(meta1, meta2) is False


def test_validate_metadata_vocab_index_mismatch():
    """Verify that different vocabulary INDICES for same keys cause failure (strict check)."""
    meta1 = create_basis_metadata(smile_vocabulary={"A": 1, "B": 2})
    meta2 = create_basis_metadata(smile_vocabulary={"A": 2, "B": 1})
    assert TrainingDataSerializer.validate_metadata(meta1, meta2) is False


def test_validate_metadata_vocab_empty_vs_none():
    """Verify that empty vocabulary behaves consistently vs None (Pydantic usually handles None as defaults)."""
    meta1 = create_basis_metadata(smile_vocabulary={})
    meta2 = create_basis_metadata(smile_vocabulary={})
    # Both empty dicts -> Compatible
    assert TrainingDataSerializer.validate_metadata(meta1, meta2) is True


def test_validate_metadata_normalization_stats():
    """Verify that normalization stats differences cause failure."""
    meta1 = create_basis_metadata(normalization_stats={"mean": 0.0})
    meta2 = create_basis_metadata(normalization_stats={"mean": 0.1})
    assert TrainingDataSerializer.validate_metadata(meta1, meta2) is False


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
