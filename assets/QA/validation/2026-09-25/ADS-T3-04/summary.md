# ADS-T3-04 — Public Data PubChem resolution

- **Validated:** 2026-09-25
- **Application revision:** `7e956838b041ced5169ad9b7927d720b0f9ac46a`
- **Status:** `PASS` for positive resolution, normalized persistence, provenance, reload, and secondary-endpoint degradation.

The rendered Chemicals workflow resolved the stable positive query CID `280` to Carbon Dioxide. The detail view showed formula `CO2`, molecular weight `44.009 g/mol`, InChIKey `CURLTUGMZLYLDI-UHFFFAOYSA-N`, InChI, SMILES, a rendered 2D structure, physicochemical properties, synonyms, and an available 3D conformer. The local record lists PubChem as its source and CID `280` as its external identifier.

After a full page reload, the Chemicals list still showed the normalized identity, formula, molecular weight, InChIKey, CID, and PubChem source. Reopening the detail view showed the stored identifiers and properties. The isolated database contains one PubChem source record for CID `280`; its associated properties carry PubChem provenance.

The focused provider tests simulate independent failures of the synonyms and 3D conformer endpoints. In both cases the primary compound resolution remains available; only the failed secondary data is omitted. The test suite passed 17 tests.

Evidence: [`browser-state.md`](browser-state.md), [`pubchem-record.json`](pubchem-record.json), and [`../ADS-T3-03/focused-backend-final.log`](../ADS-T3-03/focused-backend-final.log).
