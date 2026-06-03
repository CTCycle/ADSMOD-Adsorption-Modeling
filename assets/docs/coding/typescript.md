# ADSMOD TypeScript Rules

Last updated: 2026-06-03

## Baseline

- Frontend stack: React 18, TypeScript 5, and Vite 6.
- Keep strict typing behavior aligned with the frontend `tsconfig.json` files.

## Types And Contracts

- Type component props, exported functions, and service return shapes.
- Prefer `unknown` to `any` for untrusted values.
- Keep shared client contracts in the project type-definition modules used by the frontends.

## Structure

- Page-level orchestration belongs in `src/pages`.
- Reusable UI belongs in `src/components`.
- Feature-specific logic belongs in `src/features`.
- API and polling logic belongs in `src/services`.

## API Usage And State

- Call backend routes through `/api/...` endpoints.
- Reuse shared HTTP helpers such as timeout and error-extraction utilities where available.
- Preserve explicit UI states for loading, running, success, failure, and cancellation.

## Accessibility And UX Quality

- Maintain keyboard-accessible interactions where interactive non-button elements are used.
- Keep ARIA attributes aligned with current dialog, navigation, and control-label patterns.
- Preserve focus-visible styling for keyboard navigation.
