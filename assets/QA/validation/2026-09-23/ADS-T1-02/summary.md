# ADS-T1-02 — SQLite startup and restart persistence

Date: 2026-09-23

Status: `PASS`

Validated code SHA: `ed80c0975e583cd9842338fca5f59157c4071f97`

Hosted CI: [run 35852874401](https://github.com/CTCycle/ADSMOD-Adsorption-Modeling/actions/runs/35852874401)

Evidence strength: focused automated SQLite/Alembic tests + application lifespan integration + hosted CI

## Scope and result

Validated missing, empty, current, and invalid SQLite startup states; packaged
Alembic head consistency; migration concurrency and lock timeout; and record
persistence across application shutdown and restart. The application lifespan
test uses a disposable SQLite database and verifies that a dataset saved by one
app instance is readable after a second app instance initializes the same
database.

The unknown-revision scenario exposed an error-contract defect on the first
focused run: Alembic raised `CommandError` before ADSMOD could report its
`DatabaseMigrationError`. Revision lookup now translates that Alembic command
failure into the database migration error, and regression coverage confirms
the database is left unmodified. No public API or schema changes were made.

## Scenarios and evidence

| Gate | Result |
| --- | --- |
| Missing SQLite file and repeat initialization | PASS; migrations create the full schema, and the second initialization is idempotent. |
| Existing empty SQLite file | PASS; initialized to the packaged Alembic head. |
| Non-empty unversioned and empty-version-table databases | PASS; rejected without schema inference. |
| Unknown revision and incomplete schema stamped at head | PASS; both fail with `DatabaseMigrationError` and do not infer or add application tables. |
| Migration rollback, concurrent SQLite startup, and SQLite lock timeout | PASS. |
| Alembic history | PASS; exactly one packaged head, no pending operations, and the migrated database contains every ORM table. |
| Application restart persistence | PASS; dataset saved before shutdown is found after a second application lifespan starts against the same disposable database. |
| Local focused pytest command below | `22 passed` on Windows, Python 3.14.7, pytest 9.1.1. |
| Ruff on changed Python files | PASS. |
| Hosted `base-backend`, `ml-backend`, and `frontend` | PASS in run `35852874401` on the validated code SHA. |

Command:

```powershell
.\app\server\.venv\Scripts\python.exe -m pytest -c app\tests\pytest.ini app\tests\unit\test_database_initialization.py app\tests\unit\test_alembic_quality.py app\tests\persistence\test_database_restart.py app\tests\persistence\test_schema_contract.py -v --basetemp .\runtimes\cache\pytest-tmp-ads-t1-02
```

Ruff command:

```powershell
.\app\server\.venv\Scripts\python.exe -m ruff check app\server\repositories\database\migrator.py app\tests\unit\test_database_initialization.py app\tests\persistence\test_database_restart.py
```

Pytest emitted one non-blocking `PytestCacheWarning` (`WinError 183`) while
creating the configured repository cache path; all 22 tests passed. Existing
untracked/protected cache residue was preserved and excluded from the change.
Hosted CI also reported the existing Node.js 20 and future Ubuntu image
migration annotations; all jobs completed successfully.

## Remaining gap

`ADS-T1-03` route, navigation, and offline/online recovery validation remains
untested and is the next Tier 1 slice. This result makes no live PostgreSQL,
browser workflow, or ML-training claim.
