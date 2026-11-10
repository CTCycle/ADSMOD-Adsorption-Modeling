# API surface

Last updated: 2026-11-10

ADSMOD exposes one FastAPI backend. Liveness and readiness are available at
`/health/live` and `/health/ready`; there are no service-specific health
aliases or unversioned API aliases.

## Versioned API

The unified backend serves all versioned routes under `/api/v1`:

- `/system/capabilities`, `/system/configuration` and `/system/status` (backend CPU, memory including spawned training workers, database size)
- `/jobs`: read-only unified list of acquisition, fitting and training jobs
- `/datasets/*`, including the unit catalog (`GET /supported-units`) and
  user-defined units (`POST /units`, `DELETE /units/{unit_id}`). Import
  validation reports an unusable unit once as an `unsupported_unit` error. A
  unit saved without a conversion factor is a label only and is never converted.
  Dataset management: `GET /datasets` (summaries with experiment, measurement,
  material, and species counts and `last_update`), `GET /datasets/{id}`,
  `PATCH /datasets/{id}` (atomic edit change set; field-specific issues return
  HTTP 422 with `detail.issues`), `GET /datasets/{id}/delete-impact`,
  `DELETE /datasets/{id}`, `GET /datasets/{id}/experiments` (paginated with
  `offset`, `limit`, `q`, `adsorbent_id`, `adsorbate_id`, temperature range,
  `edited_only`), `GET /datasets/{id}/experiments/{experiment_id}`, and
  `GET /datasets/{id}/experiments/{experiment_id}/observations`
  (`order=pressure|source`, optional `adsorbate_id` for one species). An
  experiment lists its species as `components` (`position`, `adsorbate_id`,
  `name`); observation rows carry `adsorbate_id` and `gas_mole_fraction`.
  Edits replace a species by `position` and set `gas_mole_fraction` per
  measurement. Import mappings accept a `gas_mole_fraction` column role,
  `species_columns[].mole_fraction_column`, and `duplicate_policy`
  `reject|average`.
- `/materials` and `/species` (list, detail with the datasets that use the
  entity, and `DELETE`, which returns HTTP 409 while experiments reference it)
- `/public-data/*` (NIST, PubChem, and COD provider catalog, acquisition, and
  COD structures)
- `/fitting/*`, dataset-level fitting sessions and global model defaults:
  - `GET /models` (catalog with `enabled`, effective `lower`/`upper`,
    `canonical_lower`/`canonical_upper`, `unit`, `customized`),
    `PUT /models/{key}` (`enabled` and per-parameter bounds in canonical units,
    Pa and mol/kg), `POST /models/{key}/reset`
  - `GET /datasets/{id}/readiness` (eligibility, inverse-sigma availability,
    model incompatibility)
  - `GET /sessions` (`query`, `status[]`, `dataset_id`, `model`, `created_from`,
    `created_to`, `sort=created_at|name|status`, `direction`, `page`,
    `page_size` up to 100; returns `{items, total, page, page_size}`),
    `POST /sessions` (`name`, `dataset_id`, `models`, `optimizer`,
    `max_evaluations`, `weighting`, `parameter_overrides`, `display_units`;
    returns `session_id`, `job_id`, `poll_interval`), `GET /sessions/{id}`, and
    `DELETE /sessions/{id}` (204; 409 while running)
  - `GET /jobs/{job_id}` and `DELETE /jobs/{job_id}` for polling and cancel

  A session fits all experiments of exactly one dataset. Start returns 400 when
  any experiment is ineligible, a model is disabled, unknown, or incompatible,
  or `inverse_sigma` weighting lacks uncertainties. Global defaults replace the
  canonical `ModelSpec` bounds inside the solver (data-driven heuristics still
  apply within them); session overrides never change the globals. The effective
  bounds, overrides, experiment input hashes, and model version are stored in
  `fitting_sessions.configuration`. Sessions still `running` at startup are
  marked failed ("Interrupted by application restart"). The former
  `POST /fitting/run` and `GET /fitting/runs/{id}` are removed.
- `/analytics/*` (read-only cross-dataset queries for the dashboards:
  `GET /overview`, and `POST /selection`, `/series`, `/uptake-at-pressure`,
  `/distribution`)
- `/home/summary` (read-only totals for the Home page: datasets, experiments,
  stored species and materials, sources with at least one source record, and
  fitting sessions by status; species and materials are stored entity rows,
  the same population the Datasets page lists, whereas `/public-data/overview`
  counts provider-backed entities only)
- `/training/configuration` and the training lifecycle under `/training/*`;
  always mounted, but each request returns 503 unless deep learning is verified
- `/deep-learning/status`, `/deep-learning/plan`, `/deep-learning/operations`
  (POST, requires `confirm: true`; closes ADSMOD, runs a detached installer and
  restarts it) and `/deep-learning/log` for the optional dependency lifecycle

`/system/capabilities` is the authoritative runtime feature-discovery endpoint.
`features.machine_learning` is derived from the verified deep learning status
(real package metadata plus an import probe, cached by package fingerprint),
never from persisted state alone. The frontend uses it to expose or hide
machine learning navigation, routes and the Training settings tab.

Analytics responses are computed per request from canonical observations and
are never persisted. A response that shares a pressure axis never mixes
pressure bases (a mixed selection returns 400), every plotted point keeps its
`observation_id`, and calculated values carry a `method` (`measured`,
`nearest`, or `interpolated`). Fixed-pressure comparisons interpolate in log
pressure between bracketing points of the adsorption branch (points up to the
pressure maximum in acquisition order) and never extrapolate. Series requests
are capped at 200 series and 50,000 points and report `truncated`.

The canonical OpenAPI snapshot is `app/server/openapi/backend.json`.

The Angular client sends all API requests to the same backend origin. There is
no frontend routing split between core and machine learning services.
