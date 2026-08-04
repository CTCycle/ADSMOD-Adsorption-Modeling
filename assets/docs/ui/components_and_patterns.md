# ADSMOD Components And Patterns

Last updated: 2026-08-03

## Layout Patterns

- Console shell with a persistent sidebar, page header, and bottom service-status bar
- `datasets` page contains only custom uploaded-dataset management and import
- `public-data` page contains NIST adsorption-experiment retrieval and activity
- `public-materials` page separates NIST adsorbate and adsorbent-material retrieval, with existing PubChem enrichment
- `dashboards` page currently presents a placeholder workspace view
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
- Core navigation is routed through `app/client/src/app/app.routes.ts` and rendered by `app/client/src/app/layout/core-shell.component.ts`.
- Training sub-navigation is routed through `app/client/src/app/app.routes.ts` and rendered by `app/client/src/app/features/training/pages/machine-learning-page.component.ts`.

## Data Views And Overlays

- Sections use card and panel patterns
- Long output appears in scrollable log or markdown regions
- Dialogs use backdrop overlays and centered modal containers

## Workflow Feedback

- Long-running workflows must surface loading, running, success, error, and cancel states
- Progress indicators and logs should remain visible during background work
- Empty states should be explicit when datasets, checkpoints, or results are unavailable
