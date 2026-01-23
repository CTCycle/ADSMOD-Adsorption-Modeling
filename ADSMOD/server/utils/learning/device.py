from __future__ import annotations

from typing import Any

from ADSMOD.server.utils.logger import logger
import torch
from keras.mixed_precision import set_global_policy


class DeviceConfig:
    def __init__(self, configuration: dict[str, Any]) -> None:
        self.configuration = configuration

    # -------------------------------------------------------------------------
    def set_device(self) -> torch.device:
        use_gpu = self.configuration.get("use_device_GPU", False)
        device_name = "cuda" if use_gpu else "cpu"
        mixed_precision = self.configuration.get("use_mixed_precision", False)

        device = torch.device("cpu")
        if device_name == "cuda" and torch.cuda.is_available():
            device_id = self.configuration.get("device_ID", 0)
            device = torch.device(f"cuda:{device_id}")
            torch.cuda.set_device(device)
            logger.info("GPU (cuda:%s) is set as the active device.", device_id)
            if mixed_precision:
                set_global_policy("mixed_float16")
                logger.info("Mixed precision policy is active during training.")
        else:
            if device_name == "cuda":
                logger.info("No GPU found. Falling back to CPU.")
            logger.info("CPU is set as the active device.")

        return device
