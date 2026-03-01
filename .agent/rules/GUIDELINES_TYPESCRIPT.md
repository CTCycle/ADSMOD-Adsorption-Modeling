## TypeScript Guidelines (ADSMOD Webapp)

Project stack baseline:
- React 18 + TypeScript 5
- Vite 6 build/runtime configuration
- Strict compiler settings from `ADSMOD/client/tsconfig.json`

## 1. Type Safety Rules

- Keep `strict` mode enabled.
- Prefer `unknown` over `any` for external/untrusted values.
- Narrow values explicitly before use (type guards, `typeof`, `in`, discriminated unions).
- Type all exported functions, component props, and service layer responses.
- Use `type`/`interface` definitions in `src/types.ts` for shared API contracts.

## 2. React and State Management

- Keep components presentational when possible; move API and transformation logic to `src/services/`.
- Use `useCallback`/`useMemo` only where there is clear rerender or dependency benefit.
- Keep page-level orchestration in `src/pages/` and shared UI in `src/components/`.
- Avoid implicit `any` in event handlers and callback props.

## 3. API and Networking

- Route all HTTP calls through service modules in `src/services/`.
- Use the shared HTTP helpers (`fetchWithTimeout`, error extraction) to keep behavior consistent.
- Treat backend responses as untrusted: validate required fields before rendering.
- Keep frontend URLs relative to `API_BASE_URL` (`/api` proxy path).

## 4. Error Handling and UX

- Never swallow errors silently; return structured `{ data, error }` style results from services.
- Show actionable, user-facing status messages for async operations (start/progress/success/failure).
- Keep job polling logic centralized in service helpers rather than scattered across components.

## 5. Security

- Do not trust client inputs; backend remains source of truth.
- Avoid injecting unsanitized HTML.
- Validate and encode user-provided strings before rendering in markdown/table contexts.

## 6. Tooling and Quality Gates

- Keep `npm run build` green (`tsc && vite build`).
- Keep lint script green when lint config is present (`npm run lint`).
- Prefer small, focused modules and clear naming over abstractions.

## 7. Testing Guidance

- Frontend end-to-end behavior is primarily validated through Python Playwright tests in `tests/e2e`.
- When adding frontend behavior, update/add E2E coverage for user-visible flows.
- Keep unit-style logic isolated in service/helper functions so it is easy to test.
