# Public data architecture

Last updated: 2026-10-20

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
- `POST /sources/{source_key}/health-check` (HTTP 202)
- `GET /sync-runs`, `POST /sync-runs` (HTTP 202), `GET /sync-runs/{run_id}`
- `POST /sync-runs/{run_id}/cancel`, `GET /sync-runs/{run_id}/failures`
- `GET /adsorption` (filters: `source`, `material`, `adsorbate`, `material_id`, `adsorbate_id`, temperature range), `GET /adsorption/{isotherm_id}`
- `GET /materials`, `GET /materials/{adsorbent_id}` (with linked structures and isotherm count)
- `GET /chemicals`, `GET /chemicals/{adsorbate_id}` (with isotherm count)
- `GET /structures` (filters include `material_id`), `GET /structures/{structure_id}`
- `PUT /structures/{structure_id}/material`: explicit link or unlink (`{"adsorbent_id": null}`)
- `GET /structures/{structure_id}/file`: the stored original CIF
- `GET /structures/search`: bounded COD search

All provider acquisition runs through `POST /sync-runs`. There are no synchronous import routes and no separate NIST job API. Operations:

| Operation | Source | Parameters | Effect |
| --- | --- | --- | --- |
| `nist_index` | `nist` | `category` | Counts the records NIST publishes for one collection. |
| `nist_category` | `nist` | `category`, `fraction` | Fetches not-yet-stored records from the leading share of the index, in batches. |
| `pubchem_resolve` | `pubchem` | `query`, optional `adsorbate_id` | Resolves one compound. With `adsorbate_id`, the CID is attached to that chemical. |
| `pubchem_enrich` | `pubchem` | — | Resolves stored chemicals without PubChem identity by InChIKey or name, bounded per run. |
| `cod_import` | `cod` | `cod_id` or `cod_ids` (≤50), optional `adsorbent_id` | Imports CIF records and optionally links them to a material. |

Each run stores only the parameters its operation uses. External provider failures are recorded on the run and do not invalidate unrelated providers or local records.

The Overview endpoint returns aggregate counts and acquisition summaries from local tables in one database read. It counts provider-backed data only: isotherms and structures with a source record, plus chemicals and materials that have a source record or are used by a provider-backed isotherm. The `source` filter on Materials and Chemicals uses the same rule. `GET /sources` returns provider registration, saved health state, freshness, last successful sync time, and cached counts without contacting providers.

## Identity and deduplication

- NIST isotherm payloads carry the adsorbent hashkey, adsorbate InChIKeys, and DOI. These are attached before persistence, so isotherm species and materials resolve to the same rows as the NIST species and material catalogues. The DOI becomes a `references` row linked to the isotherm's source record, and each isotherm source record stores its per-record NIST URL.
- NIST gives some species a placeholder identifier (`NIST-ISODB-...`) instead of an InChIKey. It identifies the NIST record and the canonical row key, but it is never stored as an InChIKey.
- NIST catalogue records keep their published formula and InChI. Re-importing never clears a stored value with a missing one.
- PubChem attaches to an existing chemical only by CID link, exact InChIKey, an explicit `adsorbate_id`, or a PubChem synonym that matches exactly one stored compound. A conflicting InChIKey or a CID already linked elsewhere is rejected.
- The `20261020_public_data_nav` migration merges NIST rows created by display name into the matching NIST catalogue row. It merges only when the row has no identifier or provenance of its own, is used solely by NIST isotherms, and matches exactly one catalogue row. It also removes placeholder source records from the earlier backfill and adds indexes for the list filters.

## Durable acquisition runs

`public_data_sync_runs` stores operation parameters, queued/running/terminal status, progress counters, checkpoint, cancellation request, error summary, and timestamps. `public_data_sync_failures` stores per-record failures with an optional retry-eligible time. Source rows retain health status and last successful sync time.

The in-process coordinator polls and claims queued rows from the local database; it adds no broker or separate service. Its configured concurrency is bounded and defaults to one active run. NIST fetches are split into batches, write each batch before saving its checkpoint, and use the canonical upserts and identifier checks. With startup resume enabled, NIST and COD continue after the last checkpointed external ID. Interrupted runs remain visible; automatic resume is disabled by default. Cancellation is checked between NIST batches, PubChem enrichment records, and selected COD records.

Acquisition is incremental and bounded. NIST runs fetch at most the configured record cap and save explicit reasons for transport failures and for records excluded by unit support. PubChem enrichment records a permanent failure for compounds PubChem cannot resolve and skips them in later runs. COD preserves each returned CIF and provenance, rejects content without a CIF data block, and refuses to reassign a structure already linked to a different material.

HTTP providers read decoded bodies under `sync_response_size_limit_bytes` (64 MiB by default; the NIST isotherm index is about 18 MB). COD answers short request bursts with HTTP 403, so ADSMOD paces COD to one request per second and retries a 403 after a longer backoff. Exact COD ID lookups skip the count request.

Automatic refresh is disabled by default. It can be enabled for an explicit list of NIST collections with a configured interval and fraction; each scheduled refresh is stored as an ordinary durable run.

## Frontend workspace

The Angular route `/public-data/:view` is the single public-data workspace. Views are Overview, Acquire, Adsorption Data, Materials, Chemicals, Structures, and Sources.

- **Acquire** holds every fetch control: NIST collections (availability, share, fetch), PubChem (resolve one compound, enrich stored chemicals), and COD (search, multi-select, optional material link, import). One run tracker lists runs with progress, cancel, failed-record reasons, and a link to the imported records.
- **Domain views** are browse-only and cross-linked. Isotherms link to their material and adsorbates, materials to their structures and isotherms, and chemicals to their isotherms. Details live in a `?id=` query parameter, so every record is deep-linkable. `material_id` and `adsorbate_id` query parameters filter isotherms and structures.
- Shortcuts carry context into Acquire. "Resolve on PubChem" on a chemical attaches the result to that chemical, and "Find COD structures" on a material preselects it as the link target.
- **Sources** shows provider registration, saved health, freshness, and stored provenance counts, and queues health checks.

The central tables use bounded columns, truncation for long identifiers, explicit detail actions, server-side pagination, loading/empty/error states, and horizontally scrollable table frames when the viewport cannot show all scientific columns.

## Structural-viewer limitation

ADSMOD stores original CIF content, unit-cell information, and normalized fractional coordinates, but this change does not add a new WebGL crystallographic viewer dependency. The stored model is sufficient for a future maintained scientific viewer without another database redesign. A viewer should be added only after selecting a maintained library with acceptable bundle size, licensing, and Angular compatibility.

## External-source limitations

- NIST availability and response behavior are controlled by the external service.
- PubChem secondary enrichment endpoints, such as synonyms or 3D conformers, may be unavailable even when the primary compound record resolves successfully.
- COD search is intentionally bounded for interactive use and is not a bulk mirroring mechanism.
- ADSMOD does not scrape sources that lack an appropriate documented/public access mechanism.
- Cross-source entity resolution is conservative. Weak name/formula similarity remains visible to users but is not treated as proof of identity.
