# ADSMOD Components And Patterns

Last updated: 2026-11-09

## Layout Patterns

- Console shell with a persistent sidebar, page header, and bottom service-status bar
- `datasets` page contains only custom uploaded-dataset management and import
- `public-data` page contains NIST adsorption-experiment retrieval and activity
- `public-data/:view` is the single public-data workspace for NIST retrieval, normalized records, PubChem enrichment, and COD structures
- `dashboards` uses Overview and Explore & compare tabs; Explore pairs a faceted query sidebar with a results table, an ECharts analysis panel, and a fixed inspector drawer that closes on Escape
- Charts render through the shared `adsmod-chart` ECharts wrapper; filled markers are measurements and hollow markers are calculated values, and every chart offers its data as a table
- `modeling` uses Curve Fitting, Deep Learning (ML profile only), and Adsorption Models tabs. Curve Fitting lists sessions with search, status/dataset/model/date filters, sorting, and server-side pagination, and starts a four-step wizard (Dataset, Models, Settings, Review); the session view shows status, log, per-experiment results, and the configuration snapshot. Adsorption Models is a catalog of model cards with enable toggles and canonical-unit bound editors. Deep Learning keeps the former Training left toolbar with a right active-workspace panel
- `home` shows the Data, Sources and Modeling summary cards, then a "From source to model" strip linking Acquire, Curate and Model beside an inline SVG schematic (original artwork drawn with theme tokens, so no third-party asset or licence applies)
- Sources Catalog groups providers into collapsible categories (state persisted in `localStorage` key `adsmod.sources.collapsed`) under a compact summary card; providers show official logos from `public/sources/*.svg`

## Controls

- Buttons
  - `button.primary` for primary actions
  - `button.secondary` for bordered neutral actions
  - `ghost-button` for low-emphasis utilities
  - `.icon-button` (with `adsmod-icon`) for icon-only actions; always set `title` and `aria-label`. Dataset origins use the single map in `shared/utils/dataset-source.ts`
  - disabled states use opacity and non-interactive cursors
- Forms
  - `.select-input` and text or numeric inputs use consistent rounded corners
  - hover and focus states rely on tokenized border and surface changes

## Navigation

- Header tabs use `.header-tab` with active and hover states
- Deep Learning sub-navigation uses `.training-view-tab` with explicit active states; the Modeling tabs sit in the page header
- Core navigation is routed through `app/client/src/app/app.routes.ts` and rendered by `app/client/src/app/layout/core-shell.component.ts`.
- Modeling tabs and Deep Learning sub-navigation are routed through `app/client/src/app/app.routes.ts`; `app/client/src/app/features/modeling/modeling-page.component.ts` renders the tabs and Deep Learning is rendered by `app/client/src/app/features/training/pages/machine-learning-page.component.ts`.

## Data Views And Overlays

- Sections use card and panel patterns
- Long output appears in scrollable log or markdown regions
- Dialogs use backdrop overlays and centered modal containers
- Acquisition cards share a `.query-row` (labelled fields that wrap instead of overflowing, plus an action), a Single/Bulk segmented control, and an `.acq-stats` coverage line (label and percentage on one row, a full-width meter beneath); all are global patterns in `ui-consistency.css`. The bulk cards share one compact `.bulk-controls` row: Share input, primary action, optional quiet secondary action. Targeted NIST search is the `adsmod-nist-targeted-search` component.
- Background jobs (NIST/PubChem/COD acquisitions, model fitting, training) are monitored in the status bar's Jobs popover (`adsmod-jobs-popover`, fed by `JobsMonitorService` and `GET /api/v1/jobs`). The button shows the active count; rows expand to details, failed records, cancel/resume and result links. Polling runs at 1.5 s while jobs are active and 10 s otherwise. Catalog activity links and `sources/acquire?panel=runs` open the popover. Escape and outside clicks close it. The job list shows about ten collapsed rows (`--job-row-height`) and scrolls inside the popover; the panel is also capped to the viewport height.
- Scroll containers must not set `overscroll-behavior` on the vertical axis unless they always overflow: `contain` on a frame that cannot scroll vertically (for example a short `.data-table-frame`) stops the wheel from scrolling the page while the cursor is over it. `.data-table-frame` contains only the horizontal axis.
- The status bar also shows backend CPU, RAM and database size from `GET /api/v1/system/status` (one lightweight request per 10 s health poll; database size cached 30 s server-side).
- Mode switches use `adsmod-segmented-control` with the labels Single, Bulk (NIST Single = targeted search). Scrollbars are thin and themed globally; checkboxes are a compact 1rem custom style in `styles.css`.
- Unit fields use `adsmod-unit-combobox` (searchable, free text allowed, with an inline confirmation for unknown units)
- Experiments use `adsmod-experiment-explorer`: stacked collapsible sections with an observation table; tables collapse to labelled rows on narrow viewports instead of scrolling horizontally
- Documentation lives on the `help` route, opened from the sidebar Docs link; there is no header help button

## Workflow Feedback

- Long-running workflows must surface loading, running, success, error, and cancel states
- Progress indicators and logs should remain visible during background work
- Empty states should be explicit when datasets, checkpoints, or results are unavailable
