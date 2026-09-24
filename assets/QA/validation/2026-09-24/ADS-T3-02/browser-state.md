# NIST acquisition rendered state

Date: 2026-09-24

Route: `http://127.0.0.1:5173/public-data/sources`

Viewport: 1280×720
Browser: Codex in-app browser; official Windows launcher; isolated `%LOCALAPPDATA%` database

The browser screenshot showed the Sources acquisition table and the backend
online indicator. After reloading the page, the rendered table and activity
state reported:

| Category | Local / indexed | Last updated |
| --- | ---: | --- |
| Adsorption experiments | 26 / 39,988 | 2026-09-24 16:14 |
| Adsorbate species | 8 / 455 | 2026-09-24 16:00 |
| Adsorbent materials | 14 / 9,328 | 2026-09-24 16:00 |

The page displayed 3 categories and 48 total local records. Before the
experiment fetch, the one guest and ten host category fetches were repeated;
the second fetch for each returned zero new records. Their standalone source
links persisted, and experiment-associated entities account for the higher
guest/host entity totals shown above.

The experiment activity message reported 40 requested, 40 fetched, 26 local,
and 14 skipped for missing canonical units. The provider cards reported 37
cached NIST provenance records after the fetch. SQLite independently confirmed
26 adsorption, 1 chemical, and 10 material source records, plus 26 isotherms.

The same rendered state was visually inspected at 1280×720 after the final
reload; the category counts remained present and the backend remained online.
