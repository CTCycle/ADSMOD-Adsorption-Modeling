FROM python:3.14.0-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN pip install --no-cache-dir uv==0.8.15

COPY pyproject.toml uv.lock ./
COPY ADSMOD ./ADSMOD

RUN uv sync --frozen --no-dev

EXPOSE 8000

CMD ["sh", "-c", "uv run python -m uvicorn ADSMOD.server.app:app --host ${FASTAPI_HOST:-0.0.0.0} --port ${FASTAPI_PORT:-8000} --log-level info"]
