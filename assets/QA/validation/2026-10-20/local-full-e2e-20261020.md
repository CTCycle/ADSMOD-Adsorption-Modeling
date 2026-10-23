# Local full E2E validation — 2026-10-20

## Scope and result

Ran ADSMOD through the Windows launcher with an isolated data directory under
`assets/QA/validation/2026-10-20/local-full-e2e-20261020/`. The user's normal
ADSMOD data directory was not used. The run covered 21 capability groups:
**16 PASS, 4 PARTIAL, 1 NOT TESTED**. The core path from CSV import through
unit normalization, Langmuir fitting, persistence, and restart passed.

| # | Capability | Result | Evidence and boundary |
| --- | --- | --- | --- |
| 1 | Launcher startup, isolated data selection, stop, and relaunch | PASS | Frontend and backend became ready; imported records and provider data remained after restart. Launcher stop completed; ports 5173 and 6045 had no listeners. |
| 2 | CSV import and dataset inspection | PASS | 48 observations across four experiment groups imported and persisted. Column mapping, validation, and selected-experiment inspection worked. |
| 3 | XLS import | PASS | Six observations across two experiments imported and persisted. |
| 4 | XLSX import, metadata edit, rename, and reload | PASS | Six observations persisted; edited description/tags and renamed dataset reloaded correctly. |
| 5 | Invalid and boundary input validation | PASS | Non-numeric observations were rejected; repeated-pressure averaging produced a warning and preserved source observations. |
| 6 | Unit normalization | PASS | `0.02 bar` normalized to `2000 Pa`; `0.08189 mmol/g` normalized to `0.08189 mol/kg`. |
| 7 | Langmuir fitting | PASS | One Langmuir model fit 12 points; UI showed RMSE `2.23579e-7` and R² `1`. |
| 8 | Fit persistence and restart | PASS | Run, metrics, and parameters remained in SQLite. Recovered `qsat=5.20000004 mol/kg` and `k=7.99999970e-6 Pa^-1`, matching the synthetic input. |
| 9 | Bounded live NIST acquisition | PASS | Live syncs fetched 1 species, 10 materials, and 26 isotherms with 0 failures; 14 isotherms were already stored. |
| 10 | PubChem resolution and provenance | PASS | Resolved CO₂ to CID 280; matched the existing NIST trichloroethene record to CID 6575 and verified source provenance. |
| 11 | COD search, import, and material links | PASS | Search found COD 2300380 and 2300381; both were imported and linked to Cu-BTC. Link, unlink, and relink worked; structure details included atoms and reference metadata. |
| 12 | CIF download in the browser | PARTIAL | The browser download event timed out. The same local CIF endpoint returned HTTP 200 with 1,019,557 bytes and a valid CIF header. |
| 13 | Adsorption browse, filter, detail, and pagination | PASS | Zeolite 5A filter returned four records; detail showed six points; pagination covered rows 1–25 and 26–34. |
| 14 | Materials browse, filter, detail, and cross-links | PASS | Structure-available filter found Cu-BTC; detail showed one isotherm and two structures. |
| 15 | Chemicals browse, lookup, and detail | PASS | Public chemical properties and identifiers rendered; existing NIST/PubChem provenance was visible. |
| 16 | Structures browse, detail, and material association | PASS | Imported structures rendered with unit-cell and atom details; association changes were visible. |
| 17 | Browser back, forward, and deep-link reload | PASS | Back/forward returned to the expected training/dashboard routes; `/public-data/overview` reloaded with data intact. |
| 18 | General `/dashboards` | PARTIAL | Route and navigation worked, but it remained empty despite persisted fitting and public-data records. |
| 19 | Dataset deletion and cascade | PARTIAL | The UI confirmation did not complete through the browser automation. The isolated API delete returned HTTP 204; dataset 2, its import, 2 isotherms, and 6 observations were removed. `PRAGMA foreign_key_check` returned no violations. |
| 20 | Positive ML dataset preparation | PARTIAL | Uploaded fixture without SMILES and NIST's 486-row source were both rejected with “No valid SMILE sequences found”. All seven NIST adsorbates in that source lacked SMILES. The file chooser could not select the valid CO₂ fixture because Chrome's extension requires “Allow access to file URLs”. |
| 21 | Training run, checkpoints, and populated training dashboard | NOT TESTED | No valid processed dataset was available. Training datasets and checkpoints stayed empty; training dashboard remained idle. |

## Wizard fix and checks

The processing wizard's build summary showed the initial sample size after the
user changed it because Angular `computed()` had cached plain
`FormControl.value` reads. The value readers now return the live control values.
A focused test checks the changed value in the summary, and the rendered wizard
showed `0.1` after the same edit.

- `npm test -- --include='src/app/features/training/components/dataset-processing-wizard.component.spec.ts'` — 1 file and 2 tests passed.
- `npm run build` — passed.
- Inspected browser console — no warning or error entries in the checked tab.

## Cleanup

The launcher stopped its frontend and backend processes. Ports 5173 and 6045
were confirmed closed. The isolated database and fixture files were removed
after evidence collection. No release or production-data operation was
performed.
