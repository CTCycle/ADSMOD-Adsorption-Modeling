# Public data architecture

Last updated: 2026-11-09

## Scope

ADSMOD exposes one canonical public-data subsystem for externally sourced adsorption, chemical, material, and structural records. NIST, PubChem, and the Crystallography Open Database (COD) are providers behind the same backend boundary. The Angular client consumes normalized ADSMOD contracts and does not implement provider-specific persistence logic.

Dependency direction:

`HTTP router -> PublicDataService -> provider adapter / PublicDataRepository -> canonical database`

Provider adapters own remote protocols. The repository owns canonical persistence, source records, external identifiers, normalized properties, structures, and provenance relationships.

## Supported providers

### NIST/ARPA-E adsorption database

Purpose:

- adsorption experiments and isotherm measurements;
- adsorbate reference records;
- adsorbent/material reference records;
- durable, bounded NIST collection acquisition runs.

NIST acquisition remains asynchronous because full collection retrieval can be expensive. Retrieved experiments are persisted in the existing canonical adsorption schema and linked to NIST `source_records`.

The NIST web service is treated as an external dependency. Provider health failures do not make locally cached data unavailable.

Official references:

- https://www.nist.gov/programs-projects/nist-data-resources-adsorption
- https://adsorption.nist.gov/isodb/index.php#apis

### PubChem

Purpose:

- stable PubChem CID identity;
- preferred/IUPAC naming and synonyms;
- molecular formula and molecular weight;
- SMILES, connectivity SMILES, InChI, and InChIKey;
- selected physicochemical descriptors;
- elemental composition derived from the returned molecular formula;
- 2D structure images;
- 3D SDF conformer availability where PubChem provides one.

ADSMOD uses PubChem's documented PUG REST interface. Requests from workspace lookups, NIST enrichment, health checks, and retries share one thread-safe pacing budget below PubChem's documented five-requests-per-second ceiling. Each event loop owns its own HTTP client; only the small request-budget state is shared. Transient retries use capped exponential backoff or the provider's `Retry-After` value.

Official references:

- https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest
- https://pubchem.ncbi.nlm.nih.gov/docs/programmatic-access
- https://pubchem.ncbi.nlm.nih.gov/docs/downloads

Identity policy:

- an existing PubChem source record is authoritative for that CID;
- exact InChIKey equality may associate a PubChem record with an existing canonical adsorbate;
- display-name similarity alone never triggers an automatic merge;
- conflicting strong identifiers cause the write to be rejected rather than silently merged.

### Crystallography Open Database

Purpose:

- crystallographic records;
- CIF retrieval;
- formula and space-group metadata;
- unit-cell dimensions and angles;
- fractional atomic coordinates where a standard atom-site loop is present;
- DOI/publication metadata where supplied by COD.

COD publishes its database under CC0 and documents public query/download interfaces. ADSMOD performs a count request before an interactive search and rejects overly broad result sets rather than downloading an unbounded response. Imported structures retain the original CIF verbatim and normalize only stable unit-cell and atom-site fields.

Official references:

- https://www.crystallography.net/cod/
- https://wiki.crystallography.net/RESTful_API/
- https://wiki.crystallography.net/howtoobtaincod/
- https://creativecommons.org/publicdomain/zero/1.0/

COD records are not automatically linked to a canonical ADSMOD material by name or formula. A structure may remain unlinked until a trustworthy material association is known.

## Structure preview

- Species: the compact 2D image is PubChem's PNG for the stored CID (loaded by the browser only when a detail panel is open). The 3D conformer is cached on first use or by the bulk `conformers` run; a new PubChem resolve already downloads the SDF and caches it with no extra request. A species without a PubChem record shows no structure and links to the Acquire form rather than guessing from its name.
- Materials: only COD structures that were explicitly linked (or matched uniquely by the bulk run) are shown; the CIF is stored verbatim and rendered with 3Dmol.js. Amorphous, composite, and polymeric adsorbents usually have no crystal structure, so the panel reports none instead of an unrelated record. No other structure provider is used.
- Viewer: 3Dmol.js (BSD-3-Clause) is loaded by a dynamic import when the expanded viewer first opens and is not part of the initial bundle. Current COD files use `_space_group_symop_*` tags that 3Dmol does not read, so the viewer renames them to the legacy `_symmetry_equiv_pos_*` tags to expand the full unit cell.

## Provider interface

Every provider declares:

- a stable provider key;
- human-readable name and description;
- homepage, license, and terms links where known;
- supported capabilities;
- a bounded health check.

Health checks run as durable background jobs. The source registry reads only saved health state and marks whether that state is within the configured cache lifetime; registry and Overview requests never contact providers. The NIST probe requests only response headers for a one-byte range so it does not download a full collection index.

HTTP-backed providers use HTTPS host allowlists, bounded concurrency, request timeouts, capped response bodies, transient retries, `Retry-After` handling with a configured maximum backoff, rate-limit mapping, and a stable ADSMOD user agent. All redirects are rejected so a provider response cannot move a request outside the validated endpoint. Retry, rate-limit, and failed-request counts are logged and retained on durable run checkpoints. Provider-specific code adds only the remote query and normalization logic required by that source.

To add another source:

1. Add a provider implementing `PublicDataProvider` or `RetryingHttpProvider`.
2. Declare source capabilities and authoritative license/terms URLs.
3. Convert remote responses into normalized service/repository inputs. Do not expose the remote schema directly to the frontend.
4. Register the provider in `CoreServiceContainer`.
5. Add a source definition to `SOURCE_DEFINITIONS` and, if persistence changes, an Alembic migration.
6. Add deterministic provider-normalization and failure tests with remote calls mocked.
7. Add UI controls only for capabilities the source can reliably support.
8. Document rate limits, authentication requirements, licensing, and known limitations.

## Normalized domain and provenance model

Existing canonical entities remain authoritative for adsorption workflows:

- `datasets`
- `adsorbates`
- `adsorbents`
- `isotherms`
- `isotherm_components`
- `observations`

Public provenance is normalized separately:

- `data_sources`: one row per provider;
- `source_records`: provider records keyed uniquely by `source_id + record_type + external_id`;
- `adsorbate_source_records`: external chemical records linked to canonical adsorbates;
- `adsorbent_source_records`: external material records linked to canonical adsorbents;
- `isotherm_source_records`: external adsorption records linked to canonical isotherms;
- `structure_source_records`: external structure records linked to normalized structures;
- `references` and `source_record_references`: normalized publication relationships.

Normalized enrichment tables:

- `adsorbate_synonyms`;
- `chemical_properties`;
- `structures`;
- `structure_atoms`.

`source_records.raw_metadata` is intentionally limited to irregular source-specific metadata that does not justify a canonical column. It is not a substitute for normalized scientific fields.

## Scientific value and unit provenance

Existing observation rows continue to store both original and canonical values:

- original pressure and unit;
- canonical pressure;
- original uptake and unit;
- canonical uptake in mol/kg;
- conversion metadata.

The public-data layer does not discard those fields. Normalized views may expose canonical values for comparison, while detail views retain the source relationship and existing provenance required to understand the transformation.

## Public API

The canonical API prefix is `/api/v1/public-data`.

Key endpoints:

- `GET /overview`
- `GET /sources`
- `GET /sources/nist/collections`: stored and last-indexed available counts per NIST collection
- `GET /sources/nist/search`: targeted isotherm search by adsorbate, adsorbent, or experiment (filename or DOI) with an optional temperature range; paginated, and flags isotherms already stored. NIST has no server-side search (its index endpoints ignore query parameters), so the service joins the three downloaded indexes (isotherms, gases, materials) locally and caches them for ten minutes. Names match exactly (name, synonym, InChIKey, or hashkey) when any exact match exists, otherwise by name substring.
- `GET /chemicals/lookup`: non-persisting PubChem preview (up to five candidate compounds, flagged when already stored) used before an import
- `GET /sources/nist/edited-experiments`: number of NIST experiments edited locally, used to confirm a re-download
- `POST /sources/{source_key}/health-check` (HTTP 202)
- `GET /sync-runs`, `POST /sync-runs` (HTTP 202), `GET /sync-runs/{run_id}`
- `POST /sync-runs/{run_id}/cancel`, `GET /sync-runs/{run_id}/failures`
- `GET /structures` (filters include `material_id`), `GET /structures/{structure_id}`
- `PUT /structures/{structure_id}/material`: explicit link or unlink (`{"adsorbent_id": null}`)
- `GET /structures/{structure_id}/file`: the stored original CIF
- `GET /structures/search`: bounded COD search
- `POST /chemicals/{adsorbate_id}/conformer`: returns the species' 3D SDF; it is downloaded from PubChem once, stored as the `conformer_3d_sdf` chemical property, and served locally afterwards (404 when the species has no PubChem record or PubChem publishes no 3D conformer)

All provider acquisition runs through `POST /sync-runs`. There are no synchronous import routes and no separate NIST job API. Operations:

| Operation | Source | Parameters | Effect |
| --- | --- | --- | --- |
| `nist_index` | `nist` | `category` | Counts the records NIST publishes for one collection. It is a probe, not an acquisition: `NON_ACQUISITION_OPERATIONS` (this and `health_check`) are excluded from history/active run lists and overview counters, while the newest result still feeds collection coverage. |
| `nist_category` | `nist` | `category`, `fraction` or `identifiers` (≤200 isotherm filenames, experiments only), optional `overwrite_local_edits` | Fetches not-yet-stored records from the leading share of the index, or exactly the listed records (targeted acquisition; the compounds and materials they reference are stored first), in batches. With `overwrite_local_edits`, locally edited NIST experiments in that share are fetched again and restored from source in place (same experiment id); without it they are skipped. |
| `pubchem_resolve` | `pubchem` | `query`, optional `adsorbate_id` | Resolves one compound. With `adsorbate_id`, the CID is attached to that chemical. |
| `pubchem_enrich` | `pubchem` | optional `fraction`, optional `conformers` | Resolves the leading share of stored chemicals without PubChem identity by InChIKey or name, bounded per run. With `conformers: true` it instead downloads the 3D conformer of enriched species that have one but no cached copy; failures without a retry time are not repeated. |
| `cod_import` | `cod` | `cod_id` or `cod_ids` (≤50), optional `adsorbent_id`; or `fraction` (bulk) | Imports CIF records and optionally links them to a material. In bulk mode it searches COD by name for the leading share of stored materials that have no structure and no earlier final outcome. Only an exact, unique name match is imported and linked; no match or several candidates are recorded as skips (with candidate COD IDs) and are not retried. COD rows often have no chemical name, and one material can have many CIFs, so ambiguity is the common outcome and is resolved manually in single mode. |

Each run stores only the parameters its operation uses. External provider failures are recorded on the run and do not invalidate unrelated providers or local records.

Overview also returns local coverage for providers without a published total: `species_enriched` of `species_total` (PubChem identities over stored species) and `materials_with_structure` of `materials_total`. These are local ratios, not provider catalogue sizes.

The Overview endpoint returns aggregate counts and acquisition summaries from local tables in one database read. It counts provider-backed data only: isotherms and structures with a source record, plus chemicals and materials that have a source record or are used by a provider-backed isotherm. The `source` filter on `/api/v1/materials` and `/api/v1/species` uses the same rule. Local materials, species, and acquired isotherms are browsed on the Datasets page; acquired NIST isotherms are experiments of the `NIST ISODB` dataset. `GET /sources` returns provider registration, saved health state, freshness, last successful sync time, and cached counts without contacting providers.

## Identity and deduplication

- NIST isotherm payloads carry the adsorbent hashkey, adsorbate InChIKeys, and DOI. These are attached before persistence, so isotherm species and materials resolve to the same rows as the NIST species and material catalogues. The DOI becomes a `references` row linked to the isotherm's source record, and each isotherm source record stores its per-record NIST URL.
- NIST gives some species a placeholder identifier (`NIST-ISODB-...`) instead of an InChIKey. It identifies the NIST record and the canonical row key, but it is never stored as an InChIKey.
- NIST catalogue records keep their published formula and InChI. Re-importing never clears a stored value with a missing one.
- PubChem attaches to an existing chemical only by CID link, exact InChIKey, an explicit `adsorbate_id`, or a PubChem synonym that matches exactly one stored compound. A conflicting InChIKey or a CID already linked elsewhere is rejected.

## Durable acquisition runs

`public_data_sync_runs` stores operation parameters, queued/running/terminal status, progress counters, checkpoint, cancellation request, error summary, and timestamps. `public_data_sync_failures` stores per-record failures with an optional retry-eligible time. Source rows retain health status and last successful sync time.

The in-process coordinator polls and claims queued rows from the local database; it adds no broker or separate service. Its configured concurrency is bounded and defaults to one active run. NIST fetches are split into batches, write each batch before saving its checkpoint, and use the canonical upserts and identifier checks. With startup resume enabled, NIST and COD continue after the last checkpointed external ID. Interrupted runs remain visible; automatic resume is disabled by default. Cancellation is checked between NIST batches, PubChem enrichment records, and selected COD records.

Acquisition is incremental and bounded. NIST runs fetch at most the configured record cap and save explicit reasons for transport failures and for records excluded by unit support. PubChem enrichment records a permanent failure for compounds PubChem cannot resolve and skips them in later runs. COD preserves each returned CIF and provenance, rejects content without a CIF data block, and refuses to reassign a structure already linked to a different material.

HTTP providers read decoded bodies under `sync_response_size_limit_bytes` (64 MiB by default; the NIST isotherm index is about 18 MB). COD answers short request bursts with HTTP 403, so ADSMOD paces COD to one request per second and retries a 403 after a longer backoff. Exact COD ID lookups skip the count request. A bulk import reuses the search row for the CIF download instead of looking the entry up again, so a successful material costs three requests (count, search, CIF).

COD also drops a share of requests with `Server disconnected without sending a response` (`httpx.RemoteProtocolError`) or an empty read timeout. A live probe on 2026-11-09 measured roughly one request in five, and the rate was unchanged with keep-alive disabled (fresh connection per request) and with pacing slowed to one request per three seconds, so it is source-side instability rather than stale pooled connections or request rate. A retry sent moments later succeeds. Transport failures therefore use their own short backoff (base 0.5 s, doubling, equal jitter, capped by `retry_backoff_max_seconds`) rather than the 5 s HTTP-status backoff, and COD guarantees at least five attempts so one dropped request does not fail a record. HTTP-status retries (403, 429, 5xx) keep their longer backoff with up to 25% jitter and still honour `Retry-After`. Permanent errors (404, other 4xx, redirects, undecodable bodies) are never retried. Retry warnings and the final error name the exception class, and a record that still fails is stored as a failure with the class in its reason; it does not stop the run or discard records already imported.

Automatic refresh is disabled by default. It can be enabled for an explicit list of NIST collections with a configured interval and fraction; each scheduled refresh is stored as an ordinary durable run.

## Frontend workspace

The Angular route `/sources/:view` is the provider workspace for discovery and acquisition. Views are Catalog, Acquire, and Structures. Locally stored data is managed on the Datasets page instead.

- **Catalog** groups providers by the data type they contribute (adsorption isotherms, chemical identity and properties, crystal structures), derived from each provider's capabilities, so a new provider appears without layout changes. Each card shows description, data types, license, saved health, last acquisition, and local record counts, and the refresh icon in the tab bar queues health checks.
- **Acquire** holds every fetch control: NIST collections (availability, share, fetch), PubChem (resolve one compound, enrich stored chemicals), and COD (search, multi-select, optional material link, import). One run tracker lists runs with progress, cancel, and failed-record reasons. Fetching NIST isotherms first asks for confirmation when locally edited NIST experiments would be overwritten.
- **Structures** lists imported COD structures with a `?id=` detail panel, explicit material link and unlink, and CIF download.

The central tables use bounded columns, truncation for long identifiers, explicit detail actions, server-side pagination, loading/empty/error states, and horizontally scrollable table frames when the viewport cannot show all scientific columns.

## Structural-viewer limitation

ADSMOD stores original CIF content, unit-cell information, and normalized fractional coordinates, but this change does not add a new WebGL crystallographic viewer dependency. The stored model is sufficient for a future maintained scientific viewer without another database redesign. A viewer should be added only after selecting a maintained library with acceptable bundle size, licensing, and Angular compatibility.

## External-source limitations

- NIST availability and response behavior are controlled by the external service.
- PubChem secondary enrichment endpoints, such as synonyms or 3D conformers, may be unavailable even when the primary compound record resolves successfully.
- COD search is intentionally bounded for interactive use and is not a bulk mirroring mechanism.
- ADSMOD does not scrape sources that lack an appropriate documented/public access mechanism.
- Cross-source entity resolution is conservative. Weak name/formula similarity remains visible to users but is not treated as proof of identity.
