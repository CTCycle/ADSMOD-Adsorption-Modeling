from __future__ import annotations

from typing import Any

import numpy as np

from ADSMOD.server.common.constants import FITTING_MODEL_NAMES


###############################################################################
class AdsorptionModels:
    def __init__(self) -> None:
        self.model_names = list(FITTING_MODEL_NAMES)
        self.models = {
            "LANGMUIR": self.langmuir,
            "SIPS": self.sips,
            "FREUNDLICH": self.freundlich,
            "TEMKIN": self.temkin,
            "TOTH": self.toth,
            "DUBININ_RADUSHKEVICH": self.dubinin_radushkevich,
            "DUAL_SITE_LANGMUIR": self.dual_site_langmuir,
            "REDLICH_PETERSON": self.redlich_peterson,
            "JOVANOVIC": self.jovanovic,
        }

        missing = [name for name in self.model_names if name not in self.models]
        if missing:
            raise ValueError(f"Model definitions missing for: {', '.join(missing)}")

    # -------------------------------------------------------------------------
    @staticmethod
    def langmuir(pressure: np.ndarray, k: float, qsat: float) -> np.ndarray:
        k_p = pressure * k
        return qsat * (k_p / (1 + k_p))

    # -------------------------------------------------------------------------
    @staticmethod
    def sips(
        pressure: np.ndarray, k: float, qsat: float, exponent: float
    ) -> np.ndarray:
        k_p = k * (pressure**exponent)
        return qsat * (k_p / (1 + k_p))

    # -------------------------------------------------------------------------
    @staticmethod
    def freundlich(pressure: np.ndarray, k: float, exponent: float) -> np.ndarray:
        p = np.asarray(pressure, dtype=np.float64)
        safe_k = float(np.clip(k, 1e-12, None))
        safe_exponent = float(np.clip(exponent, 1e-12, None))
        base = np.clip(p * safe_k, 1e-12, None)
        return np.power(base, 1.0 / safe_exponent)

    # -------------------------------------------------------------------------
    @staticmethod
    def temkin(pressure: np.ndarray, k: float, beta: float) -> np.ndarray:
        p = np.asarray(pressure, dtype=np.float64)
        safe_k = float(np.clip(k, 1e-12, None))
        safe_beta = float(np.clip(beta, 1e-12, None))
        argument = np.clip(p * safe_k, 1e-12, None)
        return safe_beta * np.log(argument)

    # -------------------------------------------------------------------------
    @staticmethod
    def toth(
        pressure: np.ndarray,
        k: float,
        qsat: float,
        exponent: float,
    ) -> np.ndarray:
        p = np.asarray(pressure, dtype=np.float64)
        k_p = k * p
        k_p_n = k_p**exponent
        return qsat * k_p / (1.0 + k_p_n) ** (1.0 / exponent)

    # -------------------------------------------------------------------------
    @staticmethod
    def dubinin_radushkevich(
        pressure: np.ndarray, qsat: float, beta: float
    ) -> np.ndarray:
        p = np.asarray(pressure, dtype=np.float64)
        safe_p = np.clip(p, 1e-12, None)
        term = np.log(safe_p)
        return qsat * np.exp(-beta * term * term)

    # -------------------------------------------------------------------------
    @staticmethod
    def dual_site_langmuir(
        pressure: np.ndarray, k1: float, qsat1: float, k2: float, qsat2: float
    ) -> np.ndarray:
        p = pressure
        k1_p = k1 * p
        k2_p = k2 * p
        term1 = qsat1 * (k1_p / (1.0 + k1_p))
        term2 = qsat2 * (k2_p / (1.0 + k2_p))
        return term1 + term2

    # -------------------------------------------------------------------------
    @staticmethod
    def redlich_peterson(
        pressure: np.ndarray, k: float, a: float, beta: float
    ) -> np.ndarray:
        p = pressure
        denom = 1.0 + a * (p**beta)
        return (k * p) / denom

    # -------------------------------------------------------------------------
    @staticmethod
    def jovanovic(pressure: np.ndarray, k: float, qsat: float) -> np.ndarray:
        return qsat * (1.0 - np.exp(-k * pressure))

    # -------------------------------------------------------------------------
    def get_model(self, model_name: str) -> Any:
        normalized = (
            model_name.replace("-", "_").replace(" ", "_").upper()
            if isinstance(model_name, str)
            else model_name
        )
        try:
            return self.models[normalized]
        except KeyError as exc:
            raise ValueError(f"Model {model_name} is not supported") from exc
