# ADSMOD runtime configuration

Last updated: 2026-08-30

`app/resources/adsmod.json` is the only runtime value file. Its complete shape
is validated by `adsmod_common.config.AdsmodConfig`; the generated
`app/resources/adsmod.schema.json` is a validation aid, not a second authority.

Required sections are `version`, `runtime`, `storage`, `security`, and
`application`. The supported modes are `core` and `core-ml`.

The same config path is passed to Core, ML, the launcher, maintenance scripts,
and schema generation. Runtime hosts, ports, storage, database settings,
fitting defaults, training defaults, and polling intervals are not overridden
by frontend build-time files or ad-hoc environment configuration. The ML
process may read the internal token named by `security.internal_token_env`.

The `runtime` section supplies the Core, ML, and frontend ports. The `storage`
section supplies the root for logs, the embedded database, checkpoints, and ML
artifacts. Relative database paths are resolved below that root.
