from __future__ import annotations

import json
import os
from typing import Any

from NISTADS.app.constants import CONFIG_PATH


###############################################################################
class Configuration:
    def __init__(self) -> None:
        self.configuration = {
            # Dataset
            "seed": 42,
            "sample_size": 1.0,
            "validation_size": 0.2,
            "guest_fraction": 1.0,
            "host_fraction": 1.0,
            "experiments_fraction": 1.0,
            "parallel_tasks": 20,
            "min_measurements": 1,
            "max_measurements": 30,
            "SMILE_sequence_size": 20,
            "max_pressure": 10000,
            "max_uptake": 20.0,
            "shuffle_dataset": True,
            "shuffle_size": 256,
            # Model
            "selected_model": "SCADS Adsorption Isotherm",
            "dropout_rate": 0.2,
            "num_attention_heads": 2,
            "num_encoders": 2,
            "molecular_embedding_size": 64,
            "jit_compile": False,
            "jit_backend": "inductor",
            # Device
            "use_device_GPU": False,
            "device_id": 0,
            "use_mixed_precision": False,
            "num_workers": 0,
            # Training
            "train_seed": 42,
            "train_sample_size": 1.0,
            "split_seed": 76,
            "epochs": 100,
            "additional_epochs": 10,
            "batch_size": 32,
            "use_tensorboard": False,
            "plot_training_metrics": True,
            "save_checkpoints": False,
            "checkpoints_frequency": 1,
            # Learning rate scheduler
            "use_scheduler": False,
            "initial_LR": 0.001,
            "constant_steps": 0,
            "decay_steps": 1000,
            "target_LR": 0.0001,
            # Inference and validation
            "inference_batch_size": 32,
            "num_evaluation_samples": 10,
        }

    # -------------------------------------------------------------------------
    def get_configuration(self) -> dict[str, Any]:
        return self.configuration

    # -------------------------------------------------------------------------
    def update_value(self, key: str, value: Any) -> None:
        self.configuration[key] = value

    # -------------------------------------------------------------------------
    def save_configuration_to_json(self, name: str) -> None:
        full_path = os.path.join(CONFIG_PATH, f"{name}.json")
        with open(full_path, "w") as f:
            json.dump(self.configuration, f, indent=4)

    # -------------------------------------------------------------------------
    def load_configuration_from_json(self, name: str) -> None:
        full_path = os.path.join(CONFIG_PATH, name)
        with open(full_path) as f:
            self.configuration = json.load(f)
