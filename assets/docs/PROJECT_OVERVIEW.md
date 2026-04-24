# ADSMOD Documentation Overview

Last updated: 2026-04-24

## FILES INDEX

- PROJECT_OVERVIEW.md  
  Master index for all documentation files, plus documentation handling rules and environment rules.

- ARCHITECTURE.md  
  System architecture map: directory structure, backend/frontend modules, API endpoints, layering, persistence, and concurrency model.

- CODING_RULES.md  
  Consolidated coding standards for Python and TypeScript, including typing, validation, async guidance, and tooling expectations.

- RUNTIME_MODES.md  
  Supported runtime targets and startup procedures (local web/API, tests, and Tauri desktop), with configuration and deployment notes.

- UI_STANDARDS.md  
  Enforceable UI implementation standards derived from the current React/CSS codebase (tokens, components, responsiveness, and accessibility).

- USER_MANUAL.md  
  End-user operational guide for launching, navigating, and running the main ingestion, fitting, and training workflows.

## CONTEXT RULES

- Read documentation files only when needed for the current task.
- Defer reading until required by the task scope.
- Keep all related documents updated when behavior, structure, or conventions change.
- Always include a `Last updated: YYYY-MM-DD` line when modifying any document.
- Do not read all `SKILL.md` files indiscriminately.
- Pre-select relevant files using folder structure and user intent before opening documents.

## ENVIRONMENT RULES

- Assume Windows as the default operating environment.
- Support both CMD and PowerShell command forms when documenting operational steps.
- Prefer script-first Windows workflows already present in this repository (`.bat` launch/build/test scripts).
- Update this section whenever new environment-specific solutions or constraints are identified.
