# ui.dashboards — browser validation

Last updated: 2026-10-21

Scope: general `/dashboards` Overview and Explore & compare views, backed by
`/api/v1/analytics`. Run in the in-app browser against the local development
servers (`server.cli` on 6045, `npm run dev` on 5173) and the user's existing
SQLite database (29 isotherms, 537 measurements, 7 materials, 9 adsorbates,
3 datasets). All requests were read-only; no data was written or copied.

| Check | Result | Evidence |
| --- | --- | --- |
| Overview KPIs, temperature histogram, top materials/adsorbates, source split, spans per pressure basis | Pass; relative/absolute/partial spans are reported separately | `overview-desktop.jpg` |
| Ranking bar opens Explore filtered to that material | Pass (`?materials=<id>`) | — |
| Faceted filters keep sibling options and counts | Pass | `explore-isotherms-by-temperature.jpg` |
| Isotherms coloured by temperature (18 series, 298–383 K); log pressure axis | Pass | `explore-isotherms-by-temperature.jpg` |
| Uptake at 0.1 bar, log-pressure interpolation | Pass: 11 of 18 series bracketed, 7 reported as outside the measured range; interpolated points are hollow | `explore-uptake-at-pressure.jpg` |
| Pressure distribution with log bins | Pass (n = 365) | — |
| Inspector drawer: provenance, material, adsorbate; Escape closes | Pass | — |
| URL restores the full view after reload | Pass | — |
| 600px width: single column, no horizontal page scroll | Pass | `explore-600px.jpg` |

Not verified: the PubChem 2D structure image in the inspector, because no
stored chemical has PubChem enrichment in this database. Mixed-basis rejection,
interpolation, truncation and histogram binning are covered by
`app/tests/unit/test_analytics.py`.
