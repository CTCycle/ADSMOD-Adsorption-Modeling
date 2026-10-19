# ADSMOD frontend visual refinement

Date: 2026-10-19

## 2026-10-19 refinement pass

Retained paired captures use the same 1440x920 viewport and no populated dataset
state. The Fitting baseline includes an old run-recovery warning; its after view
is ready, so that pair compares layout rather than log contents:

- [`Fitting before and after`](../ui-polish-inspection/39-fitting-before-after.png)
- [`Custom Datasets before and after`](../ui-polish-inspection/40-datasets-before-after.png)
- [`Public Data overview before and after`](../ui-polish-inspection/41-public-data-before-after.png)
- [`Public Data hover state`](../core-visual-artifacts/public-data-layout-public--ffa79-ross-dense-scientific-views-viewport-1440x920/public-nav-unselected-hover.png)

Additional screenshots in `assets/QA/ui-polish-inspection/` cover populated and
long-name dataset cards, import preview/mapping/validation/completion, metadata
and delete dialogs, persisted observations, expanded fitting cards, all four
Training views, all six Public Data views, Help, and 600px/390px layouts. The
mapping table remains locally scrollable on phone layouts.

Validation on this checkout: frontend lint, Angular unit tests (26 files / 72
tests), production build, and preview-server tests (3/3) passed. The visual
suite passed 40 checks across the eight configured viewport projects; focused
rechecks covered phone-width import mapping and Public Data navigation hover.
The 195px viewport test is a narrow-layout proxy for 200% zoom, not a browser
zoom measurement. Narrator/Speech Recap output remains unverified.

The sample CSV was imported only into the isolated QA runtime, used for
populated screenshots, then deleted. The runtime was stopped and removed.

## Previous evidence (2026-09-30)

## Evidence

- `baseline-fitting-1280x720.jpg` records the pre-change expanded Sips card overlap.
- `after-fitting-1280x720.jpg` records the same fitting state after the card layout fix.
- `after-fitting-result-1440x920.jpg` records a completed fitting result with the result table visible.
- `after-datasets-1440x920.jpg` records the consolidated empty dataset state.
- `after-datasets-populated-1440x920.jpg` records the populated dataset row with long metadata wrapping beside its actions.
- `after-dataset-import-1440x920.jpg` records the canonical sample import preview.
- `after-public-overview-1440x920.jpg` records the corrected Public Data provider heading alignment.
- `after-materials-1440x920.jpg` records the normalized Materials filters.
- `after-shell-fitting-1024x768.jpg` records the compact narrower desktop shell.
- `after-training-900x900.jpg` records the Training Dashboard at the narrowest supported desktop viewport.

The live walkthrough used the isolated `assets/QA/visual-refinement/runtime` storage root and `app/tests/fixtures/sample_adsorption.csv`. Public Data populated/detail states not produced by the short local run remain covered by the deterministic visual fixture tests.

The base local profile did not expose populated Training metric or checkpoint-detail fixtures, so those views were checked in their live empty/idle states; existing Training component fixtures remain covered by the unit suite.

Validated viewports: 1440x920, 1280x720, 1024x768, and 900x900. The 1024x768 and 900x900 checks retain the existing compact desktop navigation arrangement; phone layouts were outside this pass.

Automated checks for the previous pass: frontend lint, Angular unit tests (26 files / 72 tests), production build, Public Data visual layout tests (3 projects), and application shell visual tests (12 tests across 1440x920, 1280x720, and 900x900).
