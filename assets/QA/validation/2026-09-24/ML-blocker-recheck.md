# ML training blocker recheck

Date: 2026-09-24
Application baseline: `396e2e092d810ef5182713a563f1de69eb0b5eca`
Current result: `BLOCKED` for a positive training run on this baseline

## Fixture and historical evidence

The repository contains
[`valid-smiles-training-fixture-2026-08-13.csv`](../../valid-smiles-training-fixture-2026-08-13.csv).
It has two atomic experiments and eight observations with CO2 represented by
`O=C=O`. The accompanying
[`2026-08-13 certification`](../../valid-smiles-training-certification-2026-08-13.md)
records a successful importer mapping, dataset build, one-epoch CPU training
run, and compatible checkpoint inspection. This is historical evidence; it
does not validate the current application baseline.

## Current environment check

The current `app/server/.venv` was queried without modifying it:

```powershell
& '.\app\server\.venv\Scripts\python.exe' -c "import importlib.util as u; print('torch=' + str(u.find_spec('torch') is not None)); print('rdkit=' + str(u.find_spec('rdkit') is not None))"
```

Output:

```text
torch=False
rdkit=False
```

The checkout is using the Base environment and lacks both required ML runtime
dependencies for the positive workflow. No training run was attempted in this
environment. A current-revision ML-profile run remains necessary before the
dataset-build, training, checkpoint, resume, and populated-dashboard gates can
be upgraded.
