# ADSMOD runtime and application settings

Last updated: 2026-10-21

## Settings ownership

The console **Settings** page at `/settings` is the authority for application behavior. Its full-model `GET /api/v1/settings`, `PUT /api/v1/settings`, and `POST /api/v1/settings/reset` endpoints read and write one typed `application_settings` row in the active database. The API returns current values, defaults, and any pending restart fields. A full `PUT` is validated as one unit and committed atomically. A missing row is seeded from the typed defaults during initialization or the next settings read.

SQLite and PostgreSQL are the supported database backends. With SQLite selected, its configured file stores both application settings and operational data. With PostgreSQL selected, the selected PostgreSQL database stores both. There is no second settings database or JSON fallback after cutover. Database credentials are kept in `database.json`; passwords are never returned by the API.

Settings take effect for future requests and new jobs after save. Long-running fitting, public-data, training, and dataset-build work uses the configuration snapshot captured when it starts. Runtime endpoint values and startup recovery behavior are shown as restart-required when saved values differ from the running process.

The **Database** tab stores only the selected backend and connection parameters in `database.json` in the ADSMOD data directory. It is read before the application database is opened. The file is schema-validated and written with an atomic replacement. Windows protects the password with DPAPI; other platforms store it in a mode-`0600` file. The API omits the password and reports only whether one is configured.

Database switching is performed in the running backend after the candidate connection and Alembic schema are checked. Switching is rejected while a job or public-data sync is active, and ordinary requests and database sessions are drained before the engine is replaced. The selected database's own settings row is loaded immediately; settings and operational data are not copied. Host and listener port changes still require a restart because they bind the backend and frontend processes.

## Complete UI mapping

| Settings page | Stored field | Control and default | Effect and restart requirement |
| --- | --- | --- | --- |
| Runtime | `runtime.host` | Loopback selector; `127.0.0.1` | Backend listener, frontend launch, and proxy target after restart. Loopback values only. **Restart.** |
| Runtime | `runtime.backend_port` | Integer input; `6045` | Backend API port after restart; valid range 1024–65535. Must differ from the frontend port. **Restart.** |
| Runtime | `runtime.frontend_port` | Integer input; `5173` | Console port after restart; valid range 1024–65535. Must differ from the backend port. **Restart.** |
| Data import | `datasets.allowed_extensions` | Checkbox group; `.csv`, `.xls`, `.xlsx` | Controls parser acceptance, upload capabilities, and the file picker immediately. At least one supported extension must remain selected. |
| Data import | `datasets.column_detection_cutoff` | Slider, 0–1; `0.6` | Low-confidence column suggestions are left for manual mapping; experiment grouping suggestions use the same threshold. Immediate. |
| Data import | `fitting.preview_row_limit` | Integer input, 1–500; `20` | Caps raw source rows returned by import preview. Immediate. |
| Public data | `nist.parallel_tasks` | Integer input, 1–100; `40` | Limits concurrent NIST index and record requests. Immediate for new requests. |
| Public data | `public_data.request_timeout_seconds` | Seconds input, greater than 0 and at most 120; `20` | Timeout for provider requests. Immediate for new requests. |
| Public data | `public_data.retry_attempts` | Integer input, 1–6; `3` | Maximum provider attempts, including the first request. Immediate for new requests. |
| Public data | `public_data.retry_backoff_max_seconds` | Seconds input, 0.1–120; `30` | Maximum delay between provider retries. Immediate for new requests. |
| Public data | `public_data.pubchem_parallel_requests` | Integer input, 1–3; `2` | PubChem request concurrency, shared through its request budget. Immediate for new requests. |
| Public data | `public_data.cod_max_interactive_results` | Integer input, 1–1000; `250` | Maximum COD matches returned by an interactive search. Immediate. |
| Public data | `public_data.health_cache_ttl_seconds` | Seconds input, 30–86400; `900` | Freshness window for provider health results. Immediate. |
| Public data | `public_data.sync_batch_size` | Integer input, 1–500; `50` | Number of NIST records processed before sync progress is checkpointed. Immediate for new sync runs. |
| Public data | `public_data.sync_max_records_per_run` | Integer input, 1–10000; `200` | Record cap for NIST, PubChem, and COD acquisition jobs. Immediate for new sync runs. |
| Public data | `public_data.sync_max_concurrent_jobs` | Integer input, 1–4; `1` | Number of public-data sync workers; worker capacity adjusts while ADSMOD is running. |
| Public data | `public_data.sync_response_size_limit_bytes` | Integer input in KiB steps, 1 KiB–256 MiB; `67108864` (64 MiB) | Maximum decoded provider response body. The NIST isotherm index is about 18 MB, so smaller limits can prevent that acquisition. Immediate for new requests. |
| Public data | `public_data.sync_resume_on_startup` | Toggle; `false` | Requeues recoverable interrupted sync jobs during startup. **Restart.** |
| Public data | `public_data.auto_sync_enabled` | Toggle; `false` | Starts or stops the scheduled NIST refresh loop. Immediate. |
| Public data | `public_data.auto_sync_categories` | Checkbox group; none selected | NIST collections for scheduled refresh: experiments, guest species, and host materials. At least one is required when automatic refresh is enabled. Immediate. |
| Public data | `public_data.auto_sync_interval_hours` | Integer input, 1–720; `24` | Time between scheduled NIST refreshes. Immediate; the current scheduler is restarted with the new interval. |
| Public data | `public_data.auto_sync_fraction` | Slider, greater than 0 through 1; `1.0` | Fraction of each selected NIST collection requested by a scheduled refresh. Immediate. |
| Fitting & jobs | `fitting.default_max_iterations` | Integer input, 10–1,000,000,000; `1000` | Default maximum evaluations in the fitting workflow. Immediate for a newly initialized fitting form. |
| Fitting & jobs | `fitting.max_iterations_upper_bound` | Integer input, 10–1,000,000,000; `1000000000` | Server-enforced maximum evaluations for submitted fitting jobs. Must be at least the default. Immediate. |
| Fitting & jobs | `fitting.default_parameter_min` | Non-negative numeric input; `0.0` | Lower default bound for model parameter controls. Model-specific physical ranges are intersected with this bound when possible. Immediate. |
| Fitting & jobs | `fitting.default_parameter_initial` | Non-negative numeric input; `1.0` | Initial estimate for new model parameter controls, clamped to each effective parameter range. Immediate. |
| Fitting & jobs | `fitting.default_parameter_max` | Non-negative numeric input; `100.0` | Upper default bound for model parameter controls. Model-specific physical ranges are intersected with this bound when possible. Immediate. |
| Fitting & jobs | `fitting.best_model_metric` | Dropdown; `aicc` | Ranks successful model fits by the selected metric, ascending for error/information criteria and descending for R² metrics. Immediate for new jobs; the selected metric is saved with each run. |
| Fitting & jobs | `jobs.polling_interval` | Seconds input, 0.1–60; `1.0` | Polling cadence returned for new jobs. Existing jobs retain their captured cadence. Immediate for new jobs. |
| Training | `training.use_jit` | Toggle; `false` | Default JIT compilation choice for future training jobs. Immediate. |
| Training | `training.jit_backend` | Dropdown; `inductor` | Torch compile backend when JIT is enabled: `inductor`, `eager`, `aot_eager`, or `cudagraphs`. Immediate for future jobs. |
| Training | `training.use_mixed_precision` | Toggle; `false` | Default mixed-precision choice for future training jobs. Immediate. |
| Training | `training.dataloader_workers` | Integer input, 0–64; `0` | Default worker count for future training jobs. Immediate. |
| Training | `training.persistent_workers` | Toggle; `false` | Keeps training data-loader workers between epochs. Requires at least one worker. Immediate for future training jobs. |

Fitting defaults are also projected through `/api/v1/system/configuration`; training defaults, including `persistent_workers`, are projected through `/api/v1/training/configuration` when the optional ML profile is available. These endpoints are read-only views of the same saved settings. The unused `insert_batch_size` control was removed because no application behavior consumed it.

## Database and path bootstrap exception

The only database-related value read from `settings/.env` or the process environment during normal startup is `ADSMOD_STORAGE_ROOT`, which locates the data directory before `database.json` can be read. Process environment values take precedence over `.env`. No database backend or connection value is loaded from the environment after the one-time legacy migration.

| Key | Default | Purpose and validation |
| --- | --- | --- |
| `ADSMOD_STORAGE_ROOT` | `%LOCALAPPDATA%/ADSMOD` (or the user local-data directory fallback) | Root containing `database.json`, the selected SQLite file when SQLite is active, logs, and training artifacts. Relative database paths resolve below this root. |

All connection fields are in `database.json`: backend; SQLite file path and connection timeout; or PostgreSQL engine, host, port, database name, username, password, TLS/CA, and connection timeout. `settings/.env.example` contains no database connection variables. Database settings can be changed in the running UI; only the storage root itself is a startup setting.

## One-time migration from legacy installations

When `database.json` is absent, first startup makes one cutover attempt. It reads legacy database values from the old environment variables or `data/adsmod.json`, writes the dedicated database file, and removes the old database keys from `.env` and the process environment after the file is saved. It then preserves an existing `application_settings` row in the selected database. If the selected target has no settings row, a legacy SQLite settings row or `data/adsmod.json` application settings are imported once; otherwise typed defaults seed the target. A later database switch never repeats this import or copies settings between stores.

The explicit importer remains available for a controlled migration:

```powershell
Push-Location .\app
& .\server\.venv\Scripts\python.exe -m scripts.import_legacy_configuration --legacy-config ..\data\adsmod.json
Pop-Location
```

This is a migration command only. Once `database.json` exists, normal startup and runtime services do not load database configuration or application settings from legacy environment variables or `data/adsmod.json`. Per-training checkpoint `configuration.json` files remain run artifacts and are unrelated to global settings.
