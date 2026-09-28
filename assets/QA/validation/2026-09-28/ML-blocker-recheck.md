# ML training blocker recheck and resolution

Date: 2026-09-28
Result: prior Base-profile blocker resolved for the current ML-enabled
validation; `ADS-T4-02` through `ADS-T4-04` now pass their stated scopes.

## Prior Base boundary

Before the positive run, the existing `app/server/.venv` dependency-state file
recorded `Development / Base`. The Base interpreter reported:

```text
python=G:\Projects\Repositories\Active projects\ADSMOD Adsorption Modeling\app\server\.venv\Scripts\python.exe
torch=False
rdkit=False
keras=False
sklearn=False
```

That boundary correctly kept the training routes unavailable. It was an
environment blocker at that time, not a product failure.

## Current ML profile

The official installer then activated the locked `Development / ML` profile.
The current interpreter reports Torch `2.10.0+cu130`, Keras `3.13.1`, and
scikit-learn `1.8.0`; Keras uses the Torch backend, CUDA is available, and the
runtime sees one `NVIDIA GeForce RTX 3060 Laptop GPU`. The current source has no
RDKit import and the ML extra does not require it, so RDKit is not an active
gate for this implementation.

The live current-revision checks then passed:

- [`ADS-T4-02`](ADS-T4-02/summary.md) built the valid-SMILES dataset and
  persisted the immutable snapshot.
- [`ADS-T4-03`](ADS-T4-03/summary.md) completed a real one-epoch run and a
  cancelled long run with terminal status evidence.
- [`ADS-T4-04`](ADS-T4-04/summary.md) created, inspected, resumed, and deleted
  a compatible checkpoint and rendered the populated training dashboard.

The focused regression selection passed 40 tests, and the ML-profile OpenAPI
regeneration matched the 56-path canonical snapshot. The only remaining
dashboard limitation is the separate top-level `/dashboards` placeholder,
tracked as `ISSUE-005`; it is not an ML dependency blocker.

The Base route-gating evidence and the current ML-profile evidence are both
summarized in [`ADS-T4-01/summary.md`](ADS-T4-01/summary.md).
