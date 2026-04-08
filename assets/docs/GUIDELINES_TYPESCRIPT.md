# TypeScript Guidelines (ADSMOD)

Last updated: 2026-04-08

Project baseline:
- React 18,
- TypeScript 5,
- Vite 6,
- strict compiler mode from `ADSMOD/client/tsconfig.json`.

## 1. Type Safety

- Keep strict TypeScript settings enabled.
- Prefer `unknown` over `any` for untrusted/external values.
- Type exported functions, component props, and service return contracts.
- Keep shared API/domain contracts in `client/src/types.ts`.

## 2. Frontend Structure

- Keep page orchestration in `client/src/pages`.
- Keep reusable UI in `client/src/components`.
- Keep feature-specific state/logic in `client/src/features`.
- Keep API and polling logic in `client/src/services`.
- Avoid scattering backend calls directly across presentation components.

## 3. API Integration Rules

- Frontend requests should target `/api/...` endpoints (same-origin model).
- Reuse shared service helpers for request timeouts and error normalization.
- Treat backend payloads as untrusted and validate required fields before rendering.

## 4. State and UX Behavior

- Keep asynchronous status explicit (starting, running, success, failure).
- Do not swallow exceptions silently; surface actionable user messages.
- Keep job status polling centralized in service utilities.

## 5. Security and Rendering

- Do not inject unsanitized HTML.
- Escape or sanitize user-provided values rendered in markdown/table contexts.
- Keep backend as the source of truth for validation and enforcement.

## 6. Quality Gates

- Build must pass: `npm run build` (`tsc && vite build`).
- Run lint only when lint configuration is present and maintained in the repo.
- Prefer small, focused updates aligned with current component/service patterns.

## 7. Testing Alignment

- User-facing frontend behavior is primarily validated through Python Playwright E2E tests in `tests/e2e`.
- For frontend behavior changes, update E2E coverage for the impacted flow.
