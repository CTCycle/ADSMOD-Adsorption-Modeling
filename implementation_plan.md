# Implementation Plan - Refactor Device Selection

## Objective
Refactor the device selection logic in `ADSMOD\server\utils\services\loader.py` to use the dedicated `DeviceConfig` class from `ADSMOD\server\utils\learning\device.py`. This centralizes device management and configuration.

## Proposed Changes

### 1. Modify `ADSMOD\server\utils\learning\device.py`
The `DeviceConfig.set_device` method currently performs side effects (logging, setting global torch/keras state) but does not return the `torch.device` object needed by the caller.

- **Update `set_device` method**:
    - Change return type hint from `None` to `torch.device`.
    - Retrieve the created `torch.device` object (for both CUDA and CPU paths).
    - Return the `torch.device` object.

### 2. Modify `ADSMOD\server\utils\services\loader.py`
Update both `SCADSDataLoader` and `SCADSAtomicDataLoader` to use `DeviceConfig`.

- **Imports**:
    - Add `from ADSMOD.server.utils.learning.device import DeviceConfig`.
- **Refactor `SCADSDataLoader.__init__`**:
    - Remove lines 196-201 (manual device logic).
    - Replace with `self.device = DeviceConfig(configuration).set_device()`.
- **Refactor `SCADSAtomicDataLoader.__init__`**:
    - Remove lines 301-306 (manual device logic).
    - Replace with `self.device = DeviceConfig(configuration).set_device()`.

## Verification
- Ensure `loader.py` imports `DeviceConfig` correctly.
- Ensure `DeviceConfig` returns a valid `torch.device` object.
- The logic remains functionally consistent regarding GPU/CPU choice, but adds the side effects (logging, mixed precision) defined in `DeviceConfig`, which aligns with the goal of using the dedicated class.
