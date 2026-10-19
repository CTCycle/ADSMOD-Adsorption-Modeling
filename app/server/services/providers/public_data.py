# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import asyncio
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlsplit

import httpx

from server.common.utils.logger import logger

###############################################################################
class ProviderCapability(StrEnum):
    ADSORPTION = "adsorption"
    MATERIALS = "materials"
    CHEMICALS = "chemicals"
    STRUCTURES = "structures"
    REFERENCES = "references"

###############################################################################
class ProviderError(RuntimeError):
    """Base class for public-data provider failures."""

###############################################################################
class ProviderNotFoundError(ProviderError):
    """Raised when a provider cannot resolve the requested record."""

###############################################################################
class ProviderUnavailableError(ProviderError):
    """Raised for transient or invalid remote-service responses."""

###############################################################################
class ProviderRateLimitError(ProviderUnavailableError):
    """Raised when a provider still throttles ADSMOD after retries."""

###############################################################################
@dataclass(frozen=True)
class ProviderHealth:
    status: str
    detail: str | None
    checked_at: datetime
    request_stats: dict[str, int] = field(default_factory=dict)

###############################################################################
class PublicDataProvider(ABC):
    key: str
    name: str
    description: str
    homepage_url: str
    license_name: str | None
    license_url: str | None
    terms_url: str | None
    capabilities: tuple[ProviderCapability, ...]

    # -------------------------------------------------------------------------
    def source_definition(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "description": self.description,
            "homepage_url": self.homepage_url,
            "license_name": self.license_name,
            "license_url": self.license_url,
            "terms_url": self.terms_url,
            "capabilities": [capability.value for capability in self.capabilities],
        }

    # -------------------------------------------------------------------------
    @abstractmethod
    async def health(self) -> ProviderHealth:
        raise NotImplementedError

    # -------------------------------------------------------------------------
    async def close(self) -> None:
        """Release provider-owned asynchronous resources."""

###############################################################################
class RetryingHttpProvider(PublicDataProvider):
    """Shared timeout, retry, and bounded-concurrency policy for public HTTP sources."""

    retry_statuses = frozenset({429, 500, 502, 503, 504})

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        parallel_requests: int,
        request_timeout_seconds: float,
        retry_attempts: int,
        allowed_hosts: set[str],
        max_response_bytes: int = 16_777_216,
        retry_backoff_max_seconds: float = 30.0,
    ) -> None:
        self._semaphore = asyncio.Semaphore(max(1, parallel_requests))
        self.request_timeout_seconds = max(0.1, float(request_timeout_seconds))
        self.max_attempts = max(1, int(retry_attempts))
        self.allowed_hosts = frozenset(host.casefold() for host in allowed_hosts)
        self.max_response_bytes = max(1024, int(max_response_bytes))
        self.retry_backoff_max_seconds = max(0.1, float(retry_backoff_max_seconds))
        self._client: httpx.AsyncClient | None = None
        self._client_loop: asyncio.AbstractEventLoop | None = None
        self._stats_lock = threading.Lock()
        self._request_stats = {
            "requests": 0,
            "retries": 0,
            "rate_limits": 0,
            "failures": 0,
        }

    # -------------------------------------------------------------------------
    def request_statistics(self) -> dict[str, int]:
        with self._stats_lock:
            return dict(self._request_stats)

    # -------------------------------------------------------------------------
    def _record_request_stat(self, key: str) -> None:
        with self._stats_lock:
            self._request_stats[key] += 1

    # -------------------------------------------------------------------------
    async def _get_client(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        if self._client is not None and self._client_loop is loop:
            return self._client

        if self._client is not None:
            await self._client.aclose()

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.request_timeout_seconds),
            follow_redirects=False,
        )
        self._client_loop = loop
        return self._client

    # -------------------------------------------------------------------------
    async def close(self) -> None:
        client = self._client
        self._client = None
        self._client_loop = None
        if client is not None:
            await client.aclose()

    # -------------------------------------------------------------------------
    async def _before_attempt(self) -> None:
        """Provider-specific pacing hook called before every remote attempt."""

    # -------------------------------------------------------------------------
    async def _request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        self._validate_request_target(url)
        merged_headers = {
            "Accept": "application/json",
            "User-Agent": "ADSMOD/3 public-data client",
            **(headers or {}),
        }
        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            await self._before_attempt()
            self._record_request_stat("requests")
            try:
                async with self._semaphore:
                    client = await self._get_client()
                    response, retry_delay = await self._request_attempt(
                        client,
                        method,
                        url,
                        params=params,
                        headers=merged_headers,
                        attempt=attempt,
                    )
                if response is not None:
                    return response
                await asyncio.sleep(retry_delay or 0)
            except ProviderError:
                self._record_request_stat("failures")
                raise
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                if attempt < self.max_attempts - 1:
                    retry_delay = self._retry_delay(None, attempt)
                    self._record_request_stat("retries")
                    logger.warning(
                        "%s transport request failed; retrying attempt %d/%d after %.2fs: %s",
                        self.name,
                        attempt + 2,
                        self.max_attempts,
                        retry_delay,
                        exc,
                    )
                    await asyncio.sleep(retry_delay)
                    continue
                self._record_request_stat("failures")
            except httpx.HTTPStatusError as exc:
                self._record_request_stat("failures")
                raise ProviderUnavailableError(
                    f"{self.name} returned HTTP {exc.response.status_code}."
                ) from exc
        if last_error is None:
            self._record_request_stat("failures")
        raise ProviderUnavailableError(
            f"{self.name} could not be reached after {self.max_attempts} attempts."
        ) from last_error

    # -------------------------------------------------------------------------
    def _validate_request_target(self, url: str) -> None:
        parsed_url = urlsplit(url)
        if (
            parsed_url.scheme != "https"
            or parsed_url.hostname is None
            or parsed_url.hostname.casefold() not in self.allowed_hosts
        ):
            raise ProviderUnavailableError(
                f"{self.name} request target is outside its configured HTTPS host allowlist."
            )

    # -------------------------------------------------------------------------
    async def _request_attempt(
        self,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None,
        headers: dict[str, str],
        attempt: int,
    ) -> tuple[httpx.Response | None, float | None]:
        async with client.stream(
            method, url, params=params, headers=headers
        ) as streamed:
            status_code = streamed.status_code
            if status_code == 404:
                raise ProviderNotFoundError(f"{self.name} record was not found.")
            if status_code in self.retry_statuses:
                return self._retryable_response(streamed, attempt)
            if 300 <= status_code < 400:
                raise ProviderUnavailableError(
                    f"{self.name} returned an unexpected redirect."
                )
            if status_code >= 400:
                raise ProviderUnavailableError(
                    f"{self.name} returned HTTP {status_code}."
                )
            content = await self._read_bounded_body(streamed)
            return (
                httpx.Response(
                    status_code=status_code,
                    headers=streamed.headers,
                    content=content,
                    request=streamed.request,
                ),
                None,
            )

    # -------------------------------------------------------------------------
    def _retryable_response(
        self, response: httpx.Response, attempt: int
    ) -> tuple[None, float | None]:
        status_code = response.status_code
        if status_code == 429:
            self._record_request_stat("rate_limits")
        if attempt >= self.max_attempts - 1:
            if status_code == 429:
                raise ProviderRateLimitError(
                    f"{self.name} rate limit was reached; try again later."
                )
            raise ProviderUnavailableError(f"{self.name} returned HTTP {status_code}.")
        delay = self._retry_delay(response.headers.get("Retry-After"), attempt)
        self._record_request_stat("retries")
        logger.warning(
            "%s request returned HTTP %d; retrying attempt %d/%d after %.2fs",
            self.name,
            status_code,
            attempt + 2,
            self.max_attempts,
            delay,
        )
        return None, delay

    # -------------------------------------------------------------------------
    async def _read_bounded_body(self, streamed: httpx.Response) -> bytes:
        length = streamed.headers.get("Content-Length")
        if length and length.isdigit() and int(length) > self.max_response_bytes:
            raise ProviderUnavailableError(
                f"{self.name} response exceeded the configured size limit."
            )
        content = bytearray()
        async for chunk in streamed.aiter_bytes():
            content.extend(chunk)
            if len(content) > self.max_response_bytes:
                raise ProviderUnavailableError(
                    f"{self.name} response exceeded the configured size limit."
                )
        return bytes(content)

    # -------------------------------------------------------------------------
    def _retry_delay(self, retry_after: str | None, attempt: int) -> float:
        delay: float | None = None
        if retry_after:
            try:
                delay = max(0.0, float(retry_after.strip()))
            except ValueError:
                try:
                    retry_at = parsedate_to_datetime(retry_after)
                    if retry_at.tzinfo is None:
                        retry_at = retry_at.replace(tzinfo=timezone.utc)
                    delay = max(
                        0.0,
                        (retry_at - datetime.now(timezone.utc)).total_seconds(),
                    )
                except (TypeError, ValueError, OverflowError):
                    delay = None
        if delay is None:
            delay = 0.5 * (2**attempt)
        return min(delay, self.retry_backoff_max_seconds)

    # -------------------------------------------------------------------------
    async def health(self) -> ProviderHealth:
        stats_before = self.request_statistics()
        try:
            await self._health_request()
        except ProviderError as exc:
            return ProviderHealth(
                status="unavailable",
                detail=str(exc),
                checked_at=datetime.now(timezone.utc),
                request_stats=self._statistics_since(stats_before),
            )
        return ProviderHealth(
            status="available",
            detail=None,
            checked_at=datetime.now(timezone.utc),
            request_stats=self._statistics_since(stats_before),
        )

    # -------------------------------------------------------------------------
    def _statistics_since(self, previous: dict[str, int]) -> dict[str, int]:
        current = self.request_statistics()
        return {key: current[key] - previous.get(key, 0) for key in current}

    # -------------------------------------------------------------------------
    @abstractmethod
    async def _health_request(self) -> None:
        raise NotImplementedError


__all__ = [
    "ProviderCapability",
    "ProviderError",
    "ProviderHealth",
    "ProviderNotFoundError",
    "ProviderRateLimitError",
    "ProviderUnavailableError",
    "PublicDataProvider",
    "RetryingHttpProvider",
]
