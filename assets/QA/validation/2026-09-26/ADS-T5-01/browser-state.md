# ADS-T5-01 rendered browser state

Browser: Codex in-app Browser
Viewport: 1280×720
Frontend: repository preview server serving the existing `app/client/dist/browser` bundle
Backend: isolated profile on `127.0.0.1:6045`

## Provider degradation

At `/public-data/overview`, the rendered page showed:

- `Crystallography Open Database` — `UNAVAILABLE`
- `NIST/ARPA-E Database of Novel and Emerging Adsorbent Materials` — `UNAVAILABLE`
- `PubChem` — `UNAVAILABLE`
- bottom status: `Backend Online`

The page remained a usable local Public Data overview and did not present the
provider failures as successful retrieval.

## Backend recovery

1. With the backend stopped, a page reload rendered an `HTTP error 502` alert,
   a `Retry` button, and `Backend Offline` in the shell status bar.
2. After the same isolated backend was restarted, activating `Retry` removed
   the error state and rendered `Backend Online` with the dataset route
   available.

These states were captured in the live in-app Browser during the validation
run; the durable route/state record is kept here alongside the API evidence.
