# ADS-T1-03 validation summary

Date: 2026-09-23
Status: PASS
Tested implementation: `6154532dd6628bb70e98fae6fa427e2d9108cd5f`

## Scope and result

Validated the existing top-level shell routes, landing and Public Data
redirects, unknown-route recovery, capability-aware Training navigation, Help
dialog keyboard/focus behavior, unavailable Docs/Settings controls, compact
shell layout, and backend status recovery. The Dashboards placeholder remains
unchanged and is tracked separately as `ISSUE-005`.

## Evidence

- `npm run test:unit`: **PASS**, 26 test files and 70 tests. Coverage includes
  route redirects and fallback, Training visibility by profile, native
  disabled controls and accessible explanations, Help focus containment and
  restoration, and Online → Offline → Online status recovery.
- `npm run lint`: **PASS**.
- `npm run build`: **PASS**.
- Official Windows launcher with the Base profile and Codex in-app Browser:
  **PASS**. The route matrix, visible content, 569 × 654 CSS-pixel viewport,
  focus interactions, disabled-control semantics, and compact-width overflow
  check are recorded in [`browser-state.md`](browser-state.md).
- Proxied `/health/ready` and `/api/v1/system/capabilities`: **HTTP 200** for
  both. Base profile correctly reports ML, training, and checkpoints disabled;
  see [`http-checks.json`](http-checks.json).
- Browser console errors and warnings: **none observed**.
- Launcher-owned frontend/backend processes were stopped after validation;
  ports 5173 and 6045 were confirmed free.
- Hosted CI: [run 35876297668](https://github.com/CTCycle/ADSMOD-Adsorption-Modeling/actions/runs/35876297668)
  on the tested implementation SHA; all three required jobs passed.

Browser screenshots were inspected in the in-app Browser, whose screenshot
interface returned in-memory captures only. The rendered-state record above
is the durable evidence for viewport appearance and interaction outcomes.
No dataset or persistent application data was modified.

## Remaining boundaries

No Docs or Settings destination was added. The existing Dashboards placeholder
and positive ML training/dashboard validation remain outside this slice.
