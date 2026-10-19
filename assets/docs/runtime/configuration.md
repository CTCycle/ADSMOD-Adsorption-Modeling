# ADSMOD runtime configuration

Last updated: 2026-10-19

`data/adsmod.json` is the default runtime value file. The launcher and
frontend development proxy accept `ADSMOD_DATA_DIR` (or the launcher's
`-DataPath` parameter) when a different configuration directory is
desired. The complete shape is validated by
`server.configurations.settings.AdsmodConfig`; the generated
`data/adsmod.schema.json` is a validation aid, not a second authority.

The configuration describes one backend runtime plus the frontend and application settings. It does not select between multiple backend services. Machine learning availability is determined by whether the optional `ml` dependency extra is installed and loadable.

The same config path is used by the unified backend, launcher, maintenance scripts, and schema generation. Runtime hosts and ports, storage, database settings, public-data request policy, fitting defaults, training defaults, and polling intervals are not overridden by frontend build-time files or service-specific environment variables.

`application.datasets.allowed_extensions` is the authoritative upload policy. The backend import engine validates uploaded filenames against this configured list, the dataset capability response exposes the same list, and the frontend file picker consumes that response. The canonical v3 defaults are `.csv`, `.xls`, and `.xlsx`.

`settings/.env.example` is an optional developer-environment template for
local tooling and IDEs. Its `ADSMOD_DATA_DIR` setting selects the
configuration directory; it does not replace or override the canonical JSON
configuration.

The `runtime` section supplies the backend and frontend network settings. The `storage` section supplies the root for logs, the embedded database, checkpoints, and optional ML artifacts. Relative database paths are resolved below that root.

## Public data

`application.public_data` owns the common external-provider request policy:

- `request_timeout_seconds`: timeout for one external request;
- `retry_attempts` and `retry_backoff_max_seconds`: bounded retry count and maximum `Retry-After` or exponential delay;
- `pubchem_parallel_requests`: per-client PubChem concurrency, constrained to a maximum of three. Workspace lookups, health checks, and NIST enrichment also share one paced request budget;
- `cod_max_interactive_results`: maximum COD matches ADSMOD will retrieve for one interactive search after the provider's count-first check;
- `health_cache_ttl_seconds`: how long a saved provider-health result is shown as fresh;
- `sync_batch_size`, `sync_max_records_per_run`, and `sync_max_concurrent_jobs`: bounded acquisition batch size, per-run NIST record cap, and coordinator concurrency;
- COD imports accept at most 50 explicitly selected identifiers per durable run;
- `sync_response_size_limit_bytes`: maximum response body read from a provider;
- `sync_resume_on_startup`: whether queued and running acquisitions are safely re-queued after a restart; disabled runs are recorded as interrupted;
- `auto_sync_enabled`, `auto_sync_categories`, `auto_sync_interval_hours`, and `auto_sync_fraction`: optional NIST-only scheduled refresh. It is off by default and requires at least one explicitly selected collection when enabled.

`application.nist.parallel_tasks` controls NIST request concurrency. `application.nist.pubchem_parallel_tasks` remains the per-client concurrency for the existing NIST enrichment contract; its requests still use the shared PubChem pacing budget.

Provider endpoints, source identifiers, licenses, and capabilities are application-owned integration metadata rather than user configuration. See `assets/docs/architecture/public_data.md` for provider and provenance architecture.
