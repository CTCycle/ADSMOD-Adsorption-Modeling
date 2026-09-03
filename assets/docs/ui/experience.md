# ADSMOD User Experience Rules

Last updated: 2026-09-03

## Page Structure

- Unified frontend routes
  - `datasets` (custom user dataset upload and management)
  - `public-data/:view` (normalized public adsorption, materials, chemicals, structures, and source management)
  - `dashboards` (current placeholder workspace view)
  - `fitting`
  - `training` when machine learning capability discovery reports it available
- Public Data sub-views
  - `Overview`
  - `Adsorption Data`
  - `Materials`
  - `Chemicals`
  - `Structures`
  - `Sources`
- Training sub-views
  - `Data Processing`
  - `Train datasets`
  - `Checkpoints`
  - `Training Dashboard`

## Workflow UX

- Core workflows are long-running and status-heavy.
- Keep behavior consistent across dataset ingestion, fitting, NIST collection, dashboards, and training.
- Preserve non-blocking UX by polling backend job endpoints for progress.
- Keep provider-specific acquisition controls inside the Public Data `Sources` view; normalized browsing belongs in the domain views.

## Responsiveness

- Current breakpoints include `1480px`, `1360px`, `1320px`, `1200px`, `1180px`, `1100px`, `1080px`, `900px`, `768px`, `760px`, `720px`, `700px`, and `600px`.
- Multi-column layouts should collapse to single-column on narrower widths.
- Wide scientific tables should remain inside their explicit horizontal scroll containers rather than expanding the document width.
- The training toolbar should shift from vertical to wrapped horizontal layouts on smaller screens.
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
