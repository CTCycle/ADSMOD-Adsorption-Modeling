"""Canonical ADSMOD machine-learning service."""

from .bootstrap import configure_environment

configure_environment()

from .app import create_app  # noqa: E402

__all__ = ["create_app"]
