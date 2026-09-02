# ADSMOD runtime configuration

Last updated: 2026-09-02

`app/resources/adsmod.json` is the only runtime value file. Its complete shape
is validated by `adsmod_common.config.AdsmodConfig`; the generated
`app/resources/adsmod.schema.json` is a validation aid, not a second authority.

The configuration describes one backend runtime plus the frontend and
application settings. It does not select between multiple backend services.
Machine learning availability is determined by whether the optional `ml`
dependency extra is installed and loadable.

The same config path is used by the unified backend, launcher, maintenance
scripts, and schema generation. Runtime hosts and ports, storage, database
settings, fitting defaults, training defaults, and polling intervals are not
overridden by frontend build-time files or service-specific environment
variables.

The `runtime` section supplies the backend and frontend network settings. The
`storage` section supplies the root for logs, the embedded database,
checkpoints, and optional ML artifacts. Relative database paths are resolved
below that root.
