# ADS-T3-05 — COD search, CIF import, linking, and re-import

- **Validated:** 2026-09-25
- **Application revision:** `7e956838b041ced5169ad9b7927d720b0f9ac46a`
- **Status:** `PASS` for the bounded live search/import and persistence workflow.

The rendered Structures workflow searched the bounded text query `wurtzite`; the live COD provider returned 68 records. COD `1011195` was selected because it had atomic coordinates. The browser imported the record, then inspection showed Zinc sulfide, space group `P 63 m c`, cell lengths 3.80/3.80/6.23 Å, angles 90/90/120°, volume 77.9 Å³, and two normalized atom sites.

The original CIF is retained in SQLite (`2,238` characters) with SHA-256 `2a59fc77cb14d2c5e324dfb77085d2a39d464e20729d50afdb4713c55200319e`. Its COD source record retains external ID `1011195`, source URL, raw provider metadata, and retrieval time. The retained source payload is [`original-cif-1011195.cif`](original-cif-1011195.cif).

The isolated NIST/material catalog had no existing Zinc sulfide, ZnS, or zincite material. To validate the existing explicit `adsorbent_id` import contract without inferring a cross-provider match, a disposable ZnS material fixture was added only to the isolated database. The import API linked COD `1011195` to that fixture, and the rendered local Structures list showed the persisted association. Re-importing from the browser without a material ID left the association intact and kept the local structure count at one.

Limitation: the explicit link was validated against the isolated fixture, not a pre-existing NIST material; no real cross-provider material match was available. The browser search-row Import action does not choose a material itself, so the explicit association was submitted through the existing API import field and then verified in the rendered UI. No 3D viewer claim is made.

Evidence: [`browser-state.md`](browser-state.md), [`cod-search.json`](cod-search.json), [`persistence-check.json`](persistence-check.json), and [`original-cif-1011195.cif`](original-cif-1011195.cif).
