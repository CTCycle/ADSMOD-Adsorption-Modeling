# ADSMOD Components And Patterns

Last updated: 2026-06-05

## Layout Patterns

- Sticky top header with tab navigation
- `source` page uses a two-column panel layout
- `fitting` page uses controls plus status or log columns and model card grids
- `training` uses a left toolbar with a right active-workspace panel

## Controls

- Buttons
  - `button.primary` for primary actions
  - `button.secondary` for bordered neutral actions
  - `ghost-button` for low-emphasis utilities
  - disabled states use opacity and non-interactive cursors
- Forms
  - `.select-input` and text or numeric inputs use consistent rounded corners
  - hover and focus states rely on tokenized border and surface changes

## Navigation

- Header tabs use `.header-tab` with active and hover states
- Training sub-navigation uses `.training-view-tab` with explicit active states
- Core navigation is routed through `app/client/src/app/app.routes.ts` and rendered by `app/client/src/app/layout/header-tabs.component.ts`.
- Training sub-navigation is routed through `app/ml_client/src/app/app.routes.ts` and rendered by `app/ml_client/src/app/features/training/pages/machine-learning-page.component.ts`.

## Data Views And Overlays

- Sections use card and panel patterns
- Long output appears in scrollable log or markdown regions
- Dialogs use backdrop overlays and centered modal containers

## Workflow Feedback

- Long-running workflows must surface loading, running, success, error, and cancel states
- Progress indicators and logs should remain visible during background work
- Empty states should be explicit when datasets, checkpoints, or results are unavailable
