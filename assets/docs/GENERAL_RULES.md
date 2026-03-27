# General Rules

This file is the baseline for all work in this repository.  
If any rule here conflicts with another local doc, update the docs so they are coherent and keep this file as the source of truth.

## 1. Required Documentation Review

Read `assets/docs/GENERAL_RULES.md` first for every task.  
Then read only the minimum additional docs needed:

- `assets/docs/ARCHITECTURE.md` for structure, API layout, and subsystem boundaries.
- `assets/docs/BACKGROUND_JOBS.md` for long-running jobs and polling/cancellation flows.
- `assets/docs/GUIDELINES_PYTHON.md` when editing Python code.
- `assets/docs/GUIDELINES_TYPESCRIPT.md` when editing React/TypeScript code.
- `assets/docs/GUIDELINES_TESTS.md` when adding/updating tests.
- `assets/docs/PACKAGING_AND_RUNTIME_MODES.md` for launcher/runtime/packaging behavior.
- `assets/docs/README_WRITING.md` when creating or revising README content.

## 2. Runtime and Command Rules

- Shell default in this repository is PowerShell.
- Use local repository runtimes:
  - Python from `runtimes/.venv` when available.
  - Node/npm from local project/runtime setup (launcher-managed).
- Use `cmd /c` only for `.bat` scripts or CMD-only syntax.

## 3. Engineering Baseline

- Keep changes scoped to the task; avoid incidental refactors.
- Prefer small, verifiable increments: implement, wire, validate.
- Follow existing project conventions for architecture and file organization.
- Keep behavior deterministic and testable.

## 4. Documentation Maintenance

Update `assets/docs` whenever behavior, architecture, runtime assumptions, or user workflows materially change.

When updating docs:
- keep statements factual and repository-specific,
- remove stale paths/endpoints/commands,
- keep all docs consistent with one another.
