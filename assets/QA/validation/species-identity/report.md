# Species identity validation

Last updated: 2026-11-09

Run on an isolated copy of the working database at revision `20261106_modeling_sessions`, upgraded to `20261110_species_identity`.

| Check | Result |
| --- | --- |
| Row counts across upgrade and audit (species, materials, experiments, components, observations, source records, structures) | Unchanged: 455 / 2,707 / 2,036 / 2,131 / 48,235 / 4,957 / 1 |
| Migration time | 0.12 s |
| Audit time | 1.05 s |
| Status after backfill | Species: 455 resolved. Materials: 2,704 resolved, 2 provisional, 1 generic |
| Curated InChIKeys | 22 of 22 match the stored NIST-sourced record for that species |

## The 16 near-duplicate material pairs

| Outcome | Pairs |
| --- | --- |
| Proposed for review (12) | `2,3 DhaTph` / `2,3-DhaTph`; `4PE-1P COF` / `4PE-1P-COF`; `AC F` / `ACF`; `Beulah Zap Coal` / `Beulah-zap coal`; three `BFTB` COF encoding variants; `CarboTech D 55/1.5` / `D55/1.5`; `Co-BTC` / `CoBTC`; the `Co3.bpda2` pair; `COF-366 -(OMe)2-Co` / `COF-366-(OMe)2-Co`; `COF-LZU-1` / `COF-LZU1` |
| Blocked by the guard (4) | `AgI/UiO-66 (.1)` / `(1)`; `Chlorite, 1.5nm` / `15nm`; `Cd-MOF-1` / `CDMOF-1`; `Co/Cu-BTC` / `Co@CuBTC` |

## Audit precision

Plain similarity at 0.88 raised 226 name-based proposals, nearly all real variants such as `-Me` and `-OMe`. Restricting fuzzy matching to one misspelled long word reduced the total to 17: 12 separator variants, 1 curated synonym (`Basolite A100` / `MIL-53(Al)`), and 4 fuzzy items.

Manual reading of the 4 fuzzy items: `Bitaminous A P` / `Bituminous A P` is a likely misspelling. `2,3-DhaTta` / `2,3-DhaTtab`, `[Zn4O(mipcapz)3]n` / `[Zn4O(mpcapz)3]n`, and the Dy/Gd formula pair are probably different materials. These are review proposals, and none is merged automatically.

No automatic link was made by the audit.

## Not covered

Live PubChem resolution through the new guards; provider responses are mocked in tests. PostgreSQL is covered by schema compilation only.
