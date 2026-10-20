# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from server.services.providers.cod import CODProvider
from server.services.providers.nist import NISTPublicDataProvider
from server.services.providers.pubchem import PubChemProvider
from server.services.providers.public_data import (
    ProviderCapability,
    ProviderError,
    ProviderHealth,
    ProviderNotFoundError,
    ProviderRateLimitError,
    ProviderUnavailableError,
    PublicDataProvider,
    RetryingHttpProvider,
)

# Every public source ADSMOD registers, in display order.
PROVIDER_CLASSES: tuple[type[PublicDataProvider], ...] = (
    NISTPublicDataProvider,
    PubChemProvider,
    CODProvider,
)

__all__ = [
    "CODProvider",
    "PROVIDER_CLASSES",
    "NISTPublicDataProvider",
    "ProviderCapability",
    "ProviderError",
    "ProviderHealth",
    "ProviderNotFoundError",
    "ProviderRateLimitError",
    "ProviderUnavailableError",
    "PubChemProvider",
    "PublicDataProvider",
    "RetryingHttpProvider",
]
