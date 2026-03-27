# Python Guidelines (ADSMOD)

Repository baseline for Python backend, services, and tests.

## 1. Runtime Baseline

- Target version: Python `>=3.14` (from `pyproject.toml`).
- Use the repository environment at `runtimes/.venv` when present.
- Keep imports and dependencies consistent with launcher/runtime flows.

## 2. Typing and API Contracts

- Type all public functions and non-trivial internal helpers.
- Prefer modern annotations:
  - built-in generics (`list`, `dict`, `tuple`),
  - union operator (`A | B`).
- Prefer `collections.abc` for protocol-like types (`Callable`, etc.).
- Keep FastAPI contracts schema-driven with Pydantic models from `server/domain`.

## 3. Project Structure Expectations

- API handlers in `ADSMOD/server/api`.
- Business logic in `ADSMOD/server/services`.
- Data access in `ADSMOD/server/repositories` and `repositories/queries`.
- Training runtime logic in `ADSMOD/server/learning`.

Do not move business logic into route modules unless trivial.

## 4. Job and Async Rules

- Long-running operations must run via `job_manager` (`server/services/jobs.py`).
- Keep cancellation cooperative for jobs.
- Use `async` endpoints only where awaited operations are actually non-blocking.
- Avoid CPU-heavy work directly inside request handlers.

## 5. Code Style Rules

- Follow existing codebase style in touched files.
- Keep comments concise and factual.
- Keep modules focused; avoid broad refactors during feature/bug tasks.
- Use separators/docstring patterns only when already established in nearby code.

## 6. Error Handling and Logging

- Raise `HTTPException` with clear, user-safe messages at API boundaries.
- Log detailed failures via `ADSMOD.server.common.utils.logger`.
- Normalize/shorten propagated error text for job responses where appropriate.

## 7. Validation and Security

- Validate incoming payloads with Pydantic and constrained query/path params.
- Treat uploaded/remote data as untrusted.
- Keep file/path operations constrained to expected directories and validated names.

## 8. Testing Expectations

- Follow Arrange-Act-Assert.
- Add/update tests for behavior changes in:
  - `tests/unit` for isolated logic,
  - `tests/server` or `tests/e2e` for API behavior.
- Prefer deterministic tests; avoid hidden external dependencies.
