# Public data architecture

Last updated: 2026-09-02

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
- existing ADSMOD NIST acquisition jobs.

NIST acquisition remains asynchronous because full collection retrieval can be expensive. Retrieved experiments are persisted in the existing canonical adsorption schema and linked to NIST `source_records`.

The NIST web service is treated as an external dependency. Provider health failures do not make locally cached data unavailable.

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

ADSMOD uses PubChem's documented PUG REST interface. Requests are paced below PubChem's documented five-requests-per-second ceiling, bounded by configured concurrency, and retried only for transient failures.

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
- a health check.

HTTP-backed providers share bounded concurrency, timeout handling, transient retry behavior, rate-limit mapping, and a stable ADSMOD user agent. Provider-specific code adds only the remote query and normalization logic required by that source.

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
- `material_properties`;
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

- `GET /sources`
- `GET /adsorption`
- `GET /adsorption/{isotherm_id}`
- `GET /materials`
- `GET /chemicals`
- `GET /chemicals/{adsorbate_id}`
- `POST /chemicals/resolve`
- `GET /structures`
- `GET /structures/{structure_id}`
- `GET /structures/search`
- `POST /structures/import`

List endpoints use server-side pagination and filters. External provider failures are mapped to explicit HTTP errors and do not invalidate unrelated providers or local records.

## Frontend workspace

The Angular route `/public-data/:view` is the single public-data workspace. Views are:

- Overview
- Adsorption Data
- Materials
- Chemicals
- Structures
- Sources

The old split public-materials destination is removed rather than retained as a compatibility route. NIST acquisition controls live in Sources, while normalized data exploration lives in domain views.

The central tables use bounded columns, truncation for long identifiers, explicit detail actions, server-side pagination, loading/empty/error states, and horizontally scrollable table frames when the available viewport cannot represent all scientific columns safely.

## Structural-viewer limitation

ADSMOD stores original CIF content, unit-cell information, and normalized fractional coordinates, but this change does not add a new WebGL crystallographic viewer dependency. The stored model is sufficient for a future maintained scientific viewer without another database redesign. A viewer should be added only after selecting a maintained library with acceptable bundle size, licensing, and Angular compatibility.

## External-source limitations

- NIST availability and response behavior are controlled by the external service.
- PubChem secondary enrichment endpoints, such as synonyms or 3D conformers, may be unavailable even when the primary compound record resolves successfully.
- COD search is intentionally bounded for interactive use and is not a bulk mirroring mechanism.
- ADSMOD does not scrape sources that lack an appropriate documented/public access mechanism.
- Cross-source entity resolution is conservative. Weak name/formula similarity remains visible to users but is not treated as proof of identity.
