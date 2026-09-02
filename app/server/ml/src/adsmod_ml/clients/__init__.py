"""ML client package.

Backend-to-backend HTTP clients were removed when ADSMOD moved to a single
in-process FastAPI backend. Training data access is now provided through the
shared in-process contract instead of re-exporting a core HTTP client.
"""

__all__: list[str] = []
