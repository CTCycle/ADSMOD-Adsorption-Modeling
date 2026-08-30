# Service boundaries

Last updated: 2026-08-30

```mermaid
flowchart LR
    UI[Angular client] -->|same-origin /api/v1| Core[adsmod_core]
    UI -->|same-origin /api/v1/training| ML[adsmod_ml]
    Core --> Common[adsmod_common]
    ML --> Common
    Core --> DB[(Operational database)]
    ML -->|authenticated snapshot HTTP| Core
    ML --> Artifacts[(ML manifest and checkpoints)]
```

## Ownership rules

- `adsmod_common` has no FastAPI, SQLAlchemy, or Alembic dependency.
- `adsmod_core` owns SQLAlchemy models, repositories, migrations, and all
  database initialization.
- `adsmod_ml` owns training execution and artifact persistence. Its source has
  no SQLAlchemy, Alembic, or Core-package import.
- Training input crosses the boundary only as an authenticated Core snapshot;
  ML verifies the immutable content hash before use.
- The client receives capability and configuration documents from the service
  that owns them. It does not invent fitting or training defaults.

Import-boundary tests in `app/tests/backend` enforce these rules.
