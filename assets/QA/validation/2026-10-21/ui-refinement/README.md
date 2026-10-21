# UI refinement validation — 2026-10-21

## Automated checks

- `npm run build` — passed.
- `npm run lint` — passed.
- `npm run test:unit` — 27 files and 87 tests passed. Vitest used a temporary workspace-local `TEMP`/`TMP` folder after the default Windows temp path returned `EPERM` before test execution.
- `app/server/.venv/Scripts/python.exe -m pytest -q app/tests/unit/test_public_data_sync.py app/tests/backend/test_core_routes.py` — 20 passed.
- `git diff --check` — passed; Git reported only the repository's LF-to-CRLF working-copy notices.

## Browser validation

Used the running client at `http://127.0.0.1:5173` with a temporary backend and isolated storage under `runtimes/cache/codex-uiux-validation`. Scratch records did not use or modify the user's normal database.

- Workspace datasets: 26 synthetic rows; checked 5/10/25 page sizes, page navigation, selected-row highlighting, description editing, rename and restore, and delete confirmation/cancel.
- Selected a synthetic dataset with 34 experiments and 68 observations; checked compact experiment columns, 5-row paging, experiment selection, and persisted observation inspection.
- Public Data: visited Overview, Acquire, Adsorption Data, Materials, Chemicals, Structures, and Sources. Checked bounded row previews, page-size changes, pagination, filtering, detail panels, and record cross-navigation using 59 synthetic adsorption records/materials, two chemicals, and three sources.
- At a 390px viewport, checked horizontal table scrolling, stacked filters, dataset table overflow, and the dataset Add action; reset the viewport after inspection.
- Started a small NIST adsorbate-species acquisition (fraction 0.002) against isolated storage; after navigating away it completed with 1 saved, 0 skipped, and 0 failed, and its history remained after a completed-state reload. A longer run (fraction 0.1) showed queued/running state; reloading Acquire while it was active restored the same active run at 0% without queuing another. After navigating to `/datasets` and returning, it completed at 100% with 46 saved, 1 skipped, and 0 failed.

## Limits

- The one-record NIST check is a continuity/UI check, not full-catalogue evidence.
- Browser reload during a running job and route navigation were exercised. Forced offline-mode behavior, reopening a separate browser process during an in-flight job, and backend shutdown/restart during an active acquisition were not exercised in this visual pass. Focused automated tests cover restart interruption, checkpoint/counter preservation, startup-resume policy, and explicit resume. Startup resume remains opt-in and defaults off.
- Workspace file uploads remain their existing synchronous transactional flow; this validation focused durable provider acquisition jobs.
