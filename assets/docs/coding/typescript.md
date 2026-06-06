# ADSMOD TypeScript Rules

Last updated: 2026-06-05

## Baseline

- Frontend stack: Angular standalone applications, TypeScript 5, and Angular CLI build tooling.
- Keep strict typing behavior aligned with the frontend `tsconfig.json` and `tsconfig.app.json` files.

## Types And Contracts

- Type component props, exported functions, and service return shapes.
- Prefer `unknown` to `any` for untrusted values.
- Keep shared client contracts in the project type-definition modules used by the frontends.

## Structure

- Page-level orchestration belongs in `src/app/features/**/pages` when a feature has multiple subviews.
- Reusable UI belongs in `src/app/shared/components`.
- Feature-specific logic belongs in `src/app/features`.
- API and polling logic belongs in `src/app/services` or feature-owned Angular services.
- Shared UI state belongs in signal-based stores under `src/app/core/state`.

## API Usage And State

- Call backend routes through `/api/...` endpoints.
- Reuse shared HTTP helpers such as timeout and error-extraction utilities where available.
- Preserve explicit UI states for loading, running, success, failure, and cancellation.

## Accessibility And UX Quality

- Maintain keyboard-accessible interactions where interactive non-button elements are used.
- Keep ARIA attributes aligned with current dialog, navigation, and control-label patterns.
- Preserve focus-visible styling for keyboard navigation.
