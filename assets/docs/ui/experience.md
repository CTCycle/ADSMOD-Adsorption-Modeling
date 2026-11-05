# ADSMOD User Experience Rules

Last updated: 2026-11-05

## Page Structure

- Unified frontend routes
  - `datasets` (adsorption datasets with materials and species in a secondary column; the toolbar upload icon opens the import wizard)
  - `datasets/:id` (dataset header, experiment navigator, worksheet-style measurement table, explicit edit mode)
  - `sources/:view` (external provider catalog, acquisition, and COD structures)
  - `dashboards/overview` and `dashboards/explore` (read-only data exploration over stored isotherms)
  - `modeling` (sidebar entry replacing Fitting and Training; opens Curve Fitting)
- Sources sub-views
  - `Catalog`
  - `Acquire`
  - `Structures`
- Modeling tabs
  - `Curve Fitting` (`modeling/fitting`: session list; `modeling/fitting/new`: four-step wizard; `modeling/fitting/:id`: session view)
  - `Deep Learning` (`modeling/deep-learning/:view`, only when machine learning capability discovery reports it available)
  - `Adsorption Models` (`modeling/models`: catalog and global parameter defaults)
- Deep Learning sub-views
  - `Data Processing`
  - `Train datasets`
  - `Checkpoints`
  - `Training Dashboard`

## Workflow UX

- Core workflows are long-running and status-heavy.
- Keep behavior consistent across dataset ingestion, fitting, NIST collection, dashboards, and modeling.
- Preserve non-blocking UX by polling backend job endpoints for progress.
- Keep provider acquisition on the Sources page and locally stored data on the Datasets page.
- Fitting is session-based: a session fits every experiment of one dataset, and start is blocked with an explicit reason when an experiment or model is ineligible.
- Measurement tables are read-only until Edit dataset; edits stay local until Save changes, Cancel discards them, and leaving with unsaved changes asks first.
- Large collections page on the server: experiments 50 per page, measurements 100 per page.

## Responsiveness

- Current breakpoints include `1480px`, `1360px`, `1320px`, `1200px`, `1180px`, `1100px`, `1080px`, `900px`, `768px`, `760px`, `720px`, `700px`, and `600px`.
- Multi-column layouts should collapse to single-column on narrower widths.
- Wide scientific tables should remain inside their explicit horizontal scroll containers rather than expanding the document width.
- The Deep Learning toolbar should shift from vertical to wrapped horizontal layouts on smaller screens.
- Model grids should reduce progressively to three, two, and one column layouts as width decreases.

## Accessibility

- Keyboard support is required for interactive elements, including custom row-like controls.
- Preserve `aria-label` usage for navigation and control buttons.
- Preserve dialog semantics such as `role="dialog"` and related modal ARIA attributes.
- Preserve `aria-expanded` and `aria-controls` patterns for expandable UI.
- Keep explicit `:focus-visible` outlines on controls.
- Respect `prefers-reduced-motion` behavior already defined in CSS.

## Design Principles

- Prefer consistency over one-off styling.
- Optimize for clarity and predictability in technical workflows.
- Add visual complexity only when it improves comprehension.
- Keep styling token-driven and avoid conflicting one-off style systems.
