# ADS-T3-01 rendered browser record

- Date: 2026-09-24
- Baseline: `396e2e092d810ef5182713a563f1de69eb0b5eca`
- Viewport: 1280×720
- Route: `http://127.0.0.1:5173/public-data/adsorption`
- Database: isolated backup of the configured local database
- Capture: observed in the Codex in-app browser; no screenshot file retained

## Rendered states and interactions

1. Adsorption initially displayed `1–25 of 307`; Next advanced to `26–50 of
   307`.
2. Filtering the source to NIST displayed `1–25 of 255`. Opening a NIST record
   showed its `nist` source, provider identifier
   `10.1002ADMA.201502418.ISOTHERM13`, and an Open link to the NIST adsorption
   page under the Provenance section.
3. Materials search for `carbon` displayed four matching records, including
   both NIST-sourced and locally uploaded materials.
4. Chemicals displayed 27 records; Next showed the final two records (`26–27
   of 27`) with NIST source labels visible.
5. Structures displayed `0 records` and the empty state `No imported structures
   match the current filters.`
6. The Sources view completed its initial health load and displayed COD, NIST,
   and PubChem cards, plus the cached NIST record count (433).

The visible table, filter controls, pagination, record detail, provenance, and
empty state fit the inspected viewport. Browser console telemetry was not
captured, so no console-clean claim is made.
