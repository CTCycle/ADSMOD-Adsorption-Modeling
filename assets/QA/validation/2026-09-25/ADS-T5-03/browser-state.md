# ADS-T5-03 rendered browser state

The official launcher served the current implementation at `http://127.0.0.1:5173` with the backend ready at `http://127.0.0.1:6045/health/ready`.

| Route | Result at 1280x720 | Active navigation |
| --- | --- | --- |
| `/datasets` | document/body width 1280px; no horizontal overflow | Custom Datasets |
| `/public-data/overview` | document/body width 1280px; no horizontal overflow | Public Data |
| `/dashboards` | document/body width 1280px; no horizontal overflow | Dashboards |
| `/fitting` | document/body width 1280px; no horizontal overflow | Fitting |
| `/training/processing` | Base-profile redirect to `/datasets` | Custom Datasets |

Shared geometry at 1280x720: header `112px`, main content origin aligned to header at `x=224px`, status bar `34px` ending at the viewport bottom, and `aria-current="page"` on the active primary link.

The live in-app browser screenshot and the deterministic viewport captures are stored beside this record. The Playwright visual run uses isolated API fixtures for dense content and delete/rename interaction safety; no user data was deleted during validation.
