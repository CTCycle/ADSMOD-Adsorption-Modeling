# Testing Guidelines (ADSMOD)

## 1. Test Stack and Scope

Primary test execution is Python + pytest, including:
- `tests/e2e` (Playwright/API flow tests),
- `tests/unit`,
- `tests/server`,
- `tests/backend/performance`.

## 2. Canonical Windows Runner

Run:

```cmd
tests\run_tests.bat
```

The runner:
1. uses `runtimes/.venv`,
2. validates required test dependencies (when enabled),
3. starts backend/frontend only if not already running,
4. runs `pytest tests`,
5. stops only processes it started.

## 3. Prerequisites

- Existing environment: `runtimes/.venv`.
- Optional test extras installed when needed:
  - `pytest`,
  - `pytest-playwright`,
  - `psutil`.
- Playwright browsers:

```cmd
.\runtimes\.venv\Scripts\python.exe -m playwright install
```

To provision optional deps through launcher flow, set `OPTIONAL_DEPENDENCIES=true` in `ADSMOD/settings/.env` and run `ADSMOD\start_on_windows.bat`.

## 4. Manual Test Command

```cmd
.\runtimes\.venv\Scripts\python.exe -m pytest tests -v
```

## 5. URL and Environment Resolution

Tests resolve URLs from:
- `ADSMOD/settings/.env` keys (`FASTAPI_HOST`, `FASTAPI_PORT`, `UI_HOST`, `UI_PORT`),
- optional overrides:
  - `ADSMOD_TEST_FRONTEND_URL`,
  - `ADSMOD_TEST_BACKEND_URL`.

Wildcard bind hosts (`0.0.0.0`, `::`, `[::]`) are normalized to `127.0.0.1` for client requests.

## 6. API Surface for E2E Coverage

Preferred E2E calls should target `/api/...` paths used by the frontend:

- `/api/datasets/*`
- `/api/fitting/*`
- `/api/nist/*`
- `/api/training/*`

If tests intentionally hit direct (non-`/api`) routes, keep rationale explicit.

## 7. Test Quality Expectations

- Use Arrange-Act-Assert.
- Keep tests deterministic and isolated.
- Cover success, edge, and failure behavior for changed code paths.
- Minimize heavy payloads for NIST/training tests to keep runtime practical.

## 8. Troubleshooting

- Connection issues: verify `ADSMOD/settings/.env` host/port values.
- Missing Playwright browser: run the install command above.
- Missing test packages: ensure optional dependencies are installed in `runtimes/.venv`.
- Port collisions: free configured ports or adjust `.env`.
