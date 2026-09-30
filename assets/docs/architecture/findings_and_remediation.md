# Findings and remediation

Last updated: 2026-09-30

The repository previously contained two FastAPI processes with separate
startup, health, routing, proxy, configuration, and coordination concerns.
That architecture increased packaging and lifecycle complexity and required
backend-to-backend HTTP communication for machine learning workflows.

The remediation is complete:

1. ADSMOD now has one FastAPI backend process and one backend health surface.
2. SQLAlchemy, Alembic, and the operational database remain owned by the core
   package inside that process.
3. Machine learning routes are registered in-process only when the optional ML
   package and dependencies are installed.
4. Training data crosses an explicit in-process contract instead of an
   authenticated backend-to-backend HTTP boundary.
5. The Angular client discovers ML availability through
   `/api/v1/system/capabilities` and does not expose training when unavailable.
6. The launcher, packaging metadata, tests, generated API contract, scripts,
   and documentation use the same single-backend architecture.
7. Legacy dual-service configuration, service tokens, proxy routing, entry
   points, and compatibility paths were removed rather than retained.

The bounded normal and ML-enabled workflows, provider paths, and launcher
lifecycle have since been exercised. Remaining limits are recorded in the
canonical [status ledger](../project_status_ledger.md): the general dashboard
is still a product placeholder, NIST skips unsupported measurements rather than
inferring conversions, and scientific/release claims require broader data and
longer ML runs.
