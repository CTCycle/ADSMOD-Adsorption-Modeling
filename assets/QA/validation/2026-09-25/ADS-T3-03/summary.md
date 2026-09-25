# ADS-T3-03 — NIST enrichment and unsupported measurements

- **Validated:** 2026-09-25
- **Application revision:** `7e956838b041ced5169ad9b7927d720b0f9ac46a`
- **Status:** `PASS` for the bounded workflow and the accepted skip-and-report policy. The NIST component remains `PARTIAL` for catalog coverage.

The official Windows launcher ran with an isolated `%LOCALAPPDATA%` profile and database. The rendered Sources page reported 26/39,988 experiments, 7/455 guest species, and 51/9,328 host materials. The provider jobs recorded 40 experiment rows fetched, 26 persisted, and 14 skipped because those measurements lacked a safe canonical unit/metadata basis. No conversion was inferred.

Guest enrichment completed with 7 names matched and 7 rows updated. The expanded host sample fetched 47 new records and enrichment completed with 1 of 51 names matched and 1 row updated. The current UI and provider job records show both positive enrichment paths.

The user-approved policy for `ISSUE-002` is to skip and report unsupported measurements. The 14/40 exclusion remains an explicit coverage limitation; it does not block this gate under that policy. This run does not claim full-catalog coverage.

The isolated database passed `PRAGMA integrity_check` with no foreign-key violations. The configured database SHA-256 stayed `5446E0BAB31479D86D7A4C80EA5D2349008C06B13F0CC70E97FA7284F65E7722`. The launcher-owned processes and ports were cleanly stopped and the temporary profile was removed; see [`runtime-cleanup.json`](runtime-cleanup.json).

Evidence: [`browser-state.md`](browser-state.md), [`provider-jobs.json`](provider-jobs.json), [`persistence-check.json`](persistence-check.json), and the [focused backend log](focused-backend-final.log).
