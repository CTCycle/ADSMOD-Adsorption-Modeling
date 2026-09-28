# ADS-T0-02 resource-root migration validation

Date: 2026-09-28

## Scope

The default checked-in resource directory was moved from the former nested
`resources` directory under `app/` to the repository-level `resources`
directory. The Windows launcher now defaults
to that directory and accepts an alternate directory through
`-ResourcesPath`, `-ResourcesDir`, `-ResourceDirectory`, or
`ADSMOD_RESOURCES_DIR`. The frontend development proxy and batch test runner
consume the same selection.

## Evidence

- `resources/adsmod.json`, `resources/adsmod.schema.json`, both sentinel files,
  and the existing local `database.db` are present; the former nested resource
  directory is absent.
- A tracked-code search found no remaining legacy nested-resource references.
- PowerShell parser validation passed for `start_on_windows.ps1`.
- Focused path, launcher, frontend, and configuration tests passed: **16
  passed**.
- Affected backend, persistence, cache-layout, and configuration-contract tests
  passed: **24 passed**.
- Configuration schema generation reproduced the moved schema byte-for-byte;
  the Git blob hash remained `48e6864c8884d71a31a87c0f5ec8d54d0bf16854`.
- The Node proxy loaded `resources/adsmod.json` by default and loaded the same
  file from an isolated alternate resource directory when
  `ADSMOD_RESOURCES_DIR` was set.

## Boundary

The full interactive launcher lifecycle was not rerun for this path-only
migration. The historical `ADS-T0-02` lifecycle evidence remains valid for its
recorded revision; the current launcher component stays `WORKING` until a live
launcher run covers the new default and selected alternate path.

Pre-existing protected cache residue and unrelated local data were preserved.
