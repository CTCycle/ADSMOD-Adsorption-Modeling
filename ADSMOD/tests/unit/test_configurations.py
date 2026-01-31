import pytest
from ADSMOD.server.utils.constants import CONFIGURATION_FILE
from ADSMOD.server.configurations import load_configurations
from ADSMOD.server.configurations.server import (
    TrainingSettings,
    build_training_settings,
    get_server_settings,
)

def test_json_structure_matches_settings():
    """
    Verify that keys present in configurations.json [training] section
    are correctly loaded into the TrainingSettings dataclass.
    """
    settings = get_server_settings()
    config_dict = load_configurations(CONFIGURATION_FILE)
    
    training_json = config_dict.get("training", {})
    
    # Assert all keys in JSON training section exist in TrainingSettings
    # We won't assert exact values because user might change JSON, 
    # but we check that the fields are present in the dataclass
    for key in training_json:
        assert hasattr(settings.training, key), f"TrainingSettings missing field for JSON key: {key}"

def test_config_values_are_respected():
    """
    Verify that specific values set in configurations.json are reflected in the settings object.
    Checks the default values we just added.
    """
    # Create a mock payload with known values
    mock_payload = {
        "use_jit": True,
        "jit_backend": "cudagraphs",
        "use_mixed_precision": True,
        "dataloader_workers": 4,
        "prefetch_factor": 2,
        "pin_memory": False,
        "persistent_workers": True,
        "polling_interval": 0.0,
        "plot_update_batch_interval": 7,
    }
    
    training_settings = build_training_settings(mock_payload)
    
    assert training_settings.use_jit is True
    assert training_settings.jit_backend == "cudagraphs"
    assert training_settings.use_mixed_precision is True
    assert training_settings.dataloader_workers == 4
    assert training_settings.prefetch_factor == 2
    assert training_settings.pin_memory is False
    assert training_settings.persistent_workers is True
    assert training_settings.polling_interval == 0.0
    assert training_settings.plot_update_batch_interval == 7

def test_default_fallbacks():
    """
    Verify that missing values fallback to safe defaults.
    """
    empty_payload = {}
    training_settings = build_training_settings(empty_payload)
    
    # Defaults defined in server.py
    assert training_settings.use_jit is False
    assert training_settings.jit_backend == "inductor"
    assert training_settings.use_mixed_precision is False
    assert training_settings.dataloader_workers == 0
    assert training_settings.prefetch_factor == 1
    assert training_settings.pin_memory is True
    assert training_settings.persistent_workers is False
    assert training_settings.polling_interval == 1.0
    assert training_settings.plot_update_batch_interval == 10
