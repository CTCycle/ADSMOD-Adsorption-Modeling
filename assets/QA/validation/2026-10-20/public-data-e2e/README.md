# Public-data end-to-end validation (2026-10-20)

Live run through the Angular workspace (`/public-data/acquire` and the domain views) against the FastAPI backend and the real providers. A fresh SQLite database under an isolated storage root was migrated to `20261020_public_data_nav`.

| Source | Journey driven in the UI | Outcome |
| --- | --- | --- |
| NIST | Check availability for all three collections. Fetch species (share 0.1), materials (0.01) and isotherms (0.005). | 455 / 9,328 / 39,988 records available. 46 species, 94 materials and 152 isotherms saved. 48 isotherms were excluded with explicit unit reasons. |
| PubChem | From a chemical's detail panel, choose "Resolve on PubChem" (CO, attached to the existing record). Then choose "Enrich stored chemicals". | CID 281 attached without creating a duplicate. 58 of 59 remaining chemicals enriched. One NIST placeholder identity was recorded as not found and excluded from later runs. |
| COD | From the CuBTC material, choose "Find COD structures", search "HKUST-1" (15 results), and import 2 linked to CuBTC. Unlink, then re-link through the picker. Download the CIF. | 2 structures were stored with the original CIF, atom sites and DOI. The material detail lists both, and the explicit link/unlink works. |
| Health | Refresh status in Sources. | NIST and PubChem were available. COD was unreachable at that moment; a direct `curl` also failed with a TLS connect error, so this is external. |

Files:

- `sync-runs.json`: every durable run with parameters, counts and checkpoints.
- `sources.json`: the saved health state.
- `overview.json`: the overview counts.
- `database-integrity.json`: identity and duplication checks, and provenance and enrichment counts.

Defects found and fixed during this run:

- The NIST isotherm index (about 18 MB) exceeded the 16 MiB response cap.
- NIST species formula and InChI were discarded.
- NIST placeholder identifiers were stored as InChIKeys.
- The NIST health check called a removed method.
- Overview counts excluded species and materials that are known only through isotherms.
