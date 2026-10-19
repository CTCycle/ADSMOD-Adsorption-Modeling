# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import asyncio
import re
import threading
from time import monotonic
from typing import Any
from urllib.parse import quote

from server.services.providers.public_data import (
    ProviderCapability,
    ProviderError,
    ProviderNotFoundError,
    ProviderUnavailableError,
    RetryingHttpProvider,
)


_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d+(?:\.\d+)?)?")
_INCHI_KEY = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")

###############################################################################
class PubChemRequestBudget:
    """Share PubChem pacing safely across provider instances and event loops."""

    # -------------------------------------------------------------------------
    def __init__(self, minimum_interval_seconds: float = 0.22) -> None:
        self.minimum_interval_seconds = max(0.0, float(minimum_interval_seconds))
        self._lock = threading.Lock()
        self._next_request_at = 0.0

    # -------------------------------------------------------------------------
    async def wait_turn(self) -> None:
        while True:
            with self._lock:
                now = monotonic()
                delay = self._next_request_at - now
                if delay <= 0:
                    self._next_request_at = now + self.minimum_interval_seconds
                    return
            await asyncio.sleep(delay)

###############################################################################
class PubChemProvider(RetryingHttpProvider):
    key = "pubchem"
    name = "PubChem"
    description = (
        "NIH public chemical information, identifiers, molecular descriptors, and "
        "2D/3D structure records."
    )
    homepage_url = "https://pubchem.ncbi.nlm.nih.gov/"
    license_name = "Public-domain U.S. government database; upstream record attribution applies"
    license_url = "https://pubchem.ncbi.nlm.nih.gov/docs/downloads"
    terms_url = "https://pubchem.ncbi.nlm.nih.gov/docs/programmatic-access"
    capabilities = (
        ProviderCapability.CHEMICALS,
        ProviderCapability.STRUCTURES,
        ProviderCapability.REFERENCES,
    )
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    property_names = (
        "MolecularFormula",
        "MolecularWeight",
        "SMILES",
        "ConnectivitySMILES",
        "InChI",
        "InChIKey",
        "IUPACName",
        "Title",
        "XLogP",
        "TPSA",
        "HBondDonorCount",
        "HBondAcceptorCount",
        "RotatableBondCount",
        "Complexity",
        "ExactMass",
        "MonoisotopicMass",
    )

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        parallel_requests: int = 2,
        request_timeout_seconds: float = 20.0,
        retry_attempts: int = 3,
        retry_backoff_max_seconds: float = 30.0,
        max_response_bytes: int = 16_777_216,
        request_budget: PubChemRequestBudget | None = None,
    ) -> None:
        super().__init__(
            parallel_requests=min(max(1, parallel_requests), 3),
            request_timeout_seconds=request_timeout_seconds,
            retry_attempts=retry_attempts,
            allowed_hosts={"pubchem.ncbi.nlm.nih.gov"},
            max_response_bytes=max_response_bytes,
            retry_backoff_max_seconds=retry_backoff_max_seconds,
        )
        self.request_budget = request_budget or PubChemRequestBudget()

    # -------------------------------------------------------------------------
    async def _before_attempt(self) -> None:
        # PubChem asks programmatic clients to stay below five requests/second.
        await self.request_budget.wait_turn()

    # -------------------------------------------------------------------------
    async def _health_request(self) -> None:
        await self._request(
            "GET",
            f"{self.base_url}/compound/cid/1/property/Title/JSON",
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _namespace(query: str) -> tuple[str, str]:
        normalized = query.strip()
        if normalized.isdigit():
            return "cid", normalized
        upper = normalized.upper()
        if _INCHI_KEY.fullmatch(upper):
            return "inchikey", upper
        return "name", normalized

    # -------------------------------------------------------------------------
    @staticmethod
    def _number(value: Any) -> float | None:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    # -------------------------------------------------------------------------
    @staticmethod
    def _elemental_composition(formula: str | None) -> dict[str, float]:
        if not formula:
            return {}
        composition: dict[str, float] = {}
        for symbol, raw_count in _FORMULA_TOKEN.findall(formula):
            count = float(raw_count) if raw_count else 1.0
            composition[symbol] = composition.get(symbol, 0.0) + count
        return composition

    # -------------------------------------------------------------------------
    async def resolve(self, query: str) -> dict[str, Any]:
        namespace, value = self._namespace(query)
        if not value:
            raise ValueError("PubChem query must not be empty.")
        encoded = quote(value, safe="")
        property_path = ",".join(self.property_names)
        response = await self._request(
            "GET",
            f"{self.base_url}/compound/{namespace}/{encoded}/property/{property_path}/JSON",
        )
        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderUnavailableError("PubChem returned malformed JSON.") from exc
        rows = payload.get("PropertyTable", {}).get("Properties", [])
        if not rows:
            raise ProviderNotFoundError(f"PubChem could not resolve {query!r}.")
        row = dict(rows[0])
        cid = str(row.get("CID", "")).strip()
        if not cid:
            raise ProviderNotFoundError(f"PubChem did not return a CID for {query!r}.")

        synonyms: list[str] = []
        try:
            synonym_response = await self._request(
                "GET", f"{self.base_url}/compound/cid/{cid}/synonyms/JSON"
            )
            try:
                synonym_payload = synonym_response.json()
            except ValueError as exc:
                raise ProviderUnavailableError(
                    "PubChem returned malformed JSON for synonyms."
                ) from exc
            synonym_rows = synonym_payload.get("InformationList", {}).get(
                "Information", []
            )
            if synonym_rows:
                synonyms = [
                    str(item).strip()
                    for item in synonym_rows[0].get("Synonym", [])
                    if str(item).strip()
                ][:100]
        except ProviderError:
            synonyms = []

        conformer_url = f"{self.base_url}/compound/cid/{cid}/record/SDF?record_type=3d"
        has_3d = False
        try:
            await self._request(
                "GET",
                conformer_url,
                headers={"Accept": "chemical/x-mdl-sdfile"},
            )
            has_3d = True
        except ProviderError:
            has_3d = False

        formula = str(row.get("MolecularFormula") or "").strip() or None
        descriptor_values = {
            "xlogp": self._number(row.get("XLogP")),
            "tpsa_angstrom2": self._number(row.get("TPSA")),
            "h_bond_donor_count": self._number(row.get("HBondDonorCount")),
            "h_bond_acceptor_count": self._number(row.get("HBondAcceptorCount")),
            "rotatable_bond_count": self._number(row.get("RotatableBondCount")),
            "complexity": self._number(row.get("Complexity")),
            "exact_mass_da": self._number(row.get("ExactMass")),
            "monoisotopic_mass_da": self._number(row.get("MonoisotopicMass")),
        }
        descriptors = {
            key: value for key, value in descriptor_values.items() if value is not None
        }

        return {
            "cid": cid,
            "name": str(row.get("Title") or row.get("IUPACName") or query).strip(),
            "preferred_name": str(row.get("IUPACName") or "").strip() or None,
            "formula": formula,
            "molecular_weight": self._number(row.get("MolecularWeight")),
            "smiles": str(row.get("SMILES") or "").strip() or None,
            "connectivity_smiles": str(row.get("ConnectivitySMILES") or "").strip()
            or None,
            "inchi": str(row.get("InChI") or "").strip() or None,
            "inchi_key": str(row.get("InChIKey") or "").strip() or None,
            "synonyms": synonyms,
            "descriptors": descriptors,
            "elemental_composition": self._elemental_composition(formula),
            "structure_2d_url": f"{self.base_url}/compound/cid/{cid}/PNG?image_size=300x300",
            "conformer_3d_url": conformer_url if has_3d else None,
            "source_url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
            "raw_metadata": row,
        }


__all__ = ["PubChemProvider", "PubChemRequestBudget"]
