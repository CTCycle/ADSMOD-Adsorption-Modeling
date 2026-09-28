# ADS-T5-03 rendered browser state

| Route | Viewport | Rendered observation |
| --- | --- | --- |
| `/datasets` | 1280x720 | Custom Datasets heading, Add dataset, labelled file input, Backend Online, contained shell |
| `/public-data/overview` | 1280x720 | Overview tabs, normalized workspace counts, provider coverage, Backend Online |
| `/public-data/adsorption` | 1280x720 | Source/material/adsorbate/temperature filters, table headers, no horizontal overflow |
| `/fitting` | 1280x720 | Fitting Configuration, nine model cards, fitting log, Backend Online |
| `/dashboards` | 1280x720 | `No dashboards yet` placeholder, Datasets/Public Data actions, Backend Online |
| `/datasets` | 600x900 | Narrow two-row shell navigation, readable dataset management card, Backend Online |
| `/training/processing` | 600x900 | Narrow shell and training route capture; screenshot retained for responsive proof |

The in-app Browser AX tree exposed headings, primary navigation links, dialog
names, control labels, live Backend status, and the import wizard controls. The
Codex in-app Browser surface remained at 1280x720; 600x900 uses the supporting
Playwright/Chromium screenshot command because no in-app viewport override was
available.

The import focus sequence was directly observed after uploading
`app/tests/fixtures/sample_adsorption.csv`:

```text
open:       Close import wizard
Tab:        Review mapping
Shift+Tab:  Close import wizard
Escape:     dialog absent; Add dataset focused
```

The Help dialog sequence likewise exposed `Fitting help`, `Close help`, and
`Done`; Tab moved to Done, reverse Tab wrapped to the close control, and Escape
returned focus to Help.

No audible Narrator/Speech Recap output was observable. No network trace was
exported by the in-app Browser API; no error-level console entries were
returned for the observed tab.
