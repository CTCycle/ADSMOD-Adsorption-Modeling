# UI layout regression audit — 2026-10-21

## Scope and findings

Compared the current working tree with the d312e2e console geometry changes and reviewed the dataset collection, import preview, inspector, Public Data acquisition, and record-view layouts.

- The dataset table frame declared `width: 100%` while its component added horizontal margins. The combined width exceeded the card. The shared frame now uses natural block sizing (`width: auto`), while `min-width: 0`, table minimum widths, and frame-level scrolling preserve internal scrolling for dense tables.
- NIST collection name and count styles use grid layout. Applying those classes to table cells displaced them from native table-cell layout and left the body columns misaligned with the headers. The classes now stay on inner content wrappers inside the `td` cells.
- The expanded geometry checks found no additional demonstrated overflow or sizing defect in the reviewed import preview, inspector, acquisition history, COD search, or Public Data record views.

## Validation

- Frontend lint: PASS (`npm run lint`).
- Production build: PASS (`npm run build`).
- Browser visual matrix: PASS, 88 checks across 8 viewport projects (1440×920, 1480×920, 1360×900, 1280×720, 1200×900, 900×900, 768×900, and 600×900).
- Focused geometry assertions cover dataset frame/card containment, import preview containment, inspector frames, NIST header/body alignment and desktop fill, narrow internal scrolling, and document-width containment.
- Populated rows were served through deterministic browser fixtures. The local frontend API proxy could not connect to the backend at `127.0.0.1:6045`, so this is not live-backend data verification.
- The Codex in-app browser connection closed during final verification. The rendered Playwright pages and saved screenshots were inspected in the browser; the screenshots below show the fixture-backed results.

## Screenshots

- [Workspace datasets — desktop](visual-matrix-verified/application-shell-dataset--d0a24-es-an-explicit-confirmation-viewport-1440x920/datasets-populated-desktop.png)
- [Dataset inspector — desktop](visual-matrix-verified/application-shell-dataset--d0a24-es-an-explicit-confirmation-viewport-1440x920/dataset-inspector-populated-desktop.png)
- [NIST collections — desktop](visual-matrix-verified/public-data-layout-NIST-co-7241e-nd-cooperative-cancellation-viewport-1440x920/nist-collections-populated-desktop.png)
- [NIST collections — narrow 390×844](visual-matrix-verified/public-data-layout-NIST-co-7241e-nd-cooperative-cancellation-mobile-600x900/nist-collections-narrow-390.png)
