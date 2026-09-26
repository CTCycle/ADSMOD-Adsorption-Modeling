# ADS-T5-02 rendered browser state

Browser: Codex in-app Browser
Viewport: 1280×720
Route: `/datasets`
Backend: isolated profile on `127.0.0.1:6045`

The Custom Datasets page rendered the imported record
`repeat_adsorption_6c23775c` with the `UPLOADED` badge, `2 experiments`, and
`21 observations` before restart. After the backend was stopped and restarted
twice, reloading the same route rendered the same dataset record and the shell
showed `Backend Online`.

This is rendered-state evidence of persistence, not just a direct SQLite
query. The API/database values and restart sequence are recorded in
[`restart-persistence.json`](restart-persistence.json).
