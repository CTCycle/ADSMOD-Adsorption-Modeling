# ADS-T3-03 rendered browser evidence

- Route: `http://127.0.0.1:5173/public-data/sources`
- Browser: Codex in-app Browser, 1280×720
- Backend indicator: Online
- Rendered counts after provider fetches: experiments 26/39,988; guest species 7/455; host materials 51/9,328.
- Rendered guest enrichment activity: 7 names requested, 7 matched, 7 rows updated.
- Rendered host enrichment and experiment skip counts are preserved in [`provider-jobs.json`](provider-jobs.json).

The same app build was used for the PubChem and COD workflows documented in the adjacent slice folders. Screenshot captures were inspected inline during the task. Browser security policy rejected exporting a screenshot through a generated data URL, so this file records the rendered accessibility state instead of embedding an image artifact.

Incidental follow-up: the Sources page showed a horizontal scrollbar and clipped rightmost source-card content at 1280×720. This was not a full responsive/accessibility assessment; keep `ADS-T5-03` untested and revisit it in the Tier 5 UI pass.
