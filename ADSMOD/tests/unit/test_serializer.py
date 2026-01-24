
import pytest
from ADSMOD.server.utils.repository.serializer import TrainingDataSerializer

def test_validate_metadata_match_int_str():
    """Verify that integer 20 matches string '20'."""
    metadata = {"smile_sequence_size": 20}
    target_metadata = {"smile_sequence_size": "20"}
    assert TrainingDataSerializer.validate_metadata(metadata, target_metadata) is True

def test_validate_metadata_match_str_int():
    """Verify that string '20' matches integer 20."""
    metadata = {"smile_sequence_size": "20"}
    target_metadata = {"smile_sequence_size": 20}
    assert TrainingDataSerializer.validate_metadata(metadata, target_metadata) is True

def test_validate_metadata_match_float_str():
    """Verify that float 20.0 matches string '20.0'."""
    metadata = {"smile_sequence_size": 20.0}
    target_metadata = {"smile_sequence_size": "20.0"}
    assert TrainingDataSerializer.validate_metadata(metadata, target_metadata) is True
    
def test_validate_metadata_match_float_exact():
    """Verify that float 20.0 matches 20."""
    metadata = {"smile_sequence_size": 20.0}
    target_metadata = {"smile_sequence_size": 20}
    assert TrainingDataSerializer.validate_metadata(metadata, target_metadata) is True

def test_validate_metadata_mismatch():
    """Verify that 20 does not match '21'."""
    metadata = {"smile_sequence_size": 20}
    target_metadata = {"smile_sequence_size": "21"}
    assert TrainingDataSerializer.validate_metadata(metadata, target_metadata) is False

def test_validate_metadata_critical_params():
    """Verify all critical parameters match robustly."""
    metadata = {
        "smile_vocabulary_size": 100,
        "adsorbent_vocabulary_size": 50,
        "smile_sequence_size": 20
    }
    target_metadata = {
        "smile_vocabulary_size": "100",
        "adsorbent_vocabulary_size": "50",
        "smile_sequence_size": "20"
    }
    assert TrainingDataSerializer.validate_metadata(metadata, target_metadata) is True
