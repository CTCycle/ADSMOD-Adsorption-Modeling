"""Utilities for ML training, inference, and model definitions."""

from server.services.ml_bootstrap import configure_environment

configure_environment()

__all__ = ["configure_environment"]
