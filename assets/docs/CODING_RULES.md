# ADSMOD Coding Rules

Last updated: 2026-04-24

## 1. Scope

These rules apply to all maintained source code in this repository. Keep changes scoped and avoid broad style-only churn.

## 2. Python Rules (Mandatory Baseline)

### Runtime and environment

- Target Python version: `>=3.14` (`pyproject.toml`).
- Use `runtimes/.venv` when available; fallback only to `root/.venv` if `runtimes/.venv` is absent.
- Do not create new virtual environments for normal development tasks.
- Keep dependency state aligned with `uv` and `runtimes/uv.lock`.

### Typing

- Type annotations are required for public APIs and non-trivial logic.
- Use built-in generics (`list[str]`, `dict[str, Any]`, etc.).
- Prefer `A | B` over `typing.Union`.
- Prefer `collections.abc` for abstract contracts (`Callable`, `Iterable`, etc.).
- Treat typing as a quality requirement, not optional documentation.

### Validation and API contracts

- Use Pydantic/domain models for request and response validation.
- Avoid ad-hoc manual validation for payload structures already represented by models.
- Return explicit HTTP status codes.
- Keep response models consistent and stable.
- Ensure errors are safe for clients and traceable via logs/job state.

### Async and long-running work

- Use async handlers only when dependencies are non-blocking.
- Do not run CPU-heavy workloads directly in async request handlers.
- Route long-running tasks through the existing job system (`server/services/jobs.py`).
- Long-running features must expose start, poll/status, and cancel operations.

### Code structure

- Keep functions focused and small.
- Make side effects explicit.
- Prefer simple composable logic over implicit control flow.
- Add comments only when needed for clarity or safety.
- Follow local style in touched files.
- Keep modules around or below ~1000 LOC when practical.
- Keep imports at file top.
- Avoid nested function definitions unless necessary for strict locality.
- Use classes when grouping related logic improves cohesion.

### Tooling and quality gates

- Lint/format with Ruff (or project-standard formatter/linter when configured).
- Type-check with Pylance.
- Test with pytest, including relevant `tests/unit` and `tests/e2e`/`tests/server` coverage for changed behavior.

## 3. TypeScript Rules (Inferred from Current Codebase)

### Baseline

- React 18 + TypeScript 5 + Vite 6.
- Keep strict typing behavior aligned with `ADSMOD/client/tsconfig.json`.

### Types and contracts

- Type component props, exported functions, and service return shapes.
- Prefer `unknown` to `any` for untrusted values.
- Keep shared client contracts in `client/src/types.ts`.

### Structure

- Page-level orchestration in `client/src/pages`.
- Reusable UI in `client/src/components`.
- Feature-specific logic under `client/src/features`.
- API/polling logic in `client/src/services`.

### API usage and state

- Call backend through `/api/...` endpoints.
- Reuse shared HTTP helpers (`fetchWithTimeout`, error extraction).
- Preserve explicit UI states for loading, running, success, failure, and cancellation.

### Accessibility and UX quality

- Maintain keyboard-accessible interactions (`tabIndex`, `onKeyDown`, focus-visible styles) where interactive non-button elements are used.
- Keep ARIA attributes aligned with current patterns for dialogs, navigation, and control labels.

### Build and test

- Keep `npm run build` passing (`tsc && vite build`).
- Update tests when behavior changes.

## 4. Windows Script and Operational Code

- Keep CMD (`.bat`) and PowerShell usage explicit and deterministic.
- Preserve runtime-safe path handling and avoid destructive operations outside intended directories.
- Keep script behavior compatible with the existing launcher/runtime flow.
