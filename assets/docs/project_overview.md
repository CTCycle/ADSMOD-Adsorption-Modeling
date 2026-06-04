# ADSMOD Project Overview

Last updated: 2026-06-04

## Purpose

This file is the root index for `assets/docs`. Read it first, then open the smallest topic file that answers the current question.

## Navigation Rules

1. Start with this file only.
2. Choose the relevant topic folder.
3. Open the topic overview before reading leaf documents.
4. Read only the smallest leaf file required for the task.
5. Expand to adjacent files only when the task crosses topic boundaries.

## Naming Rules

- All documentation files and folders under `assets/docs` use lowercase names.
- Root-level files are reserved for entry-point documentation only.
- Topic folders group related leaf documents by subject.

## Documentation Ontology

### Root

- [`project_overview.md`](project_overview.md)
  - Central index for the documentation tree, reading rules, and environment assumptions.

### Architecture

- [`architecture/overview.md`](architecture/overview.md)
  - Index for system structure, service boundaries, API ownership, and persistence layout.
- [`architecture.md`](architecture/architecture.md)
  - Root architecture entry point that links to the canonical architecture topic documents.
- [`architecture/system_overview.md`](architecture/system_overview.md)
  - Repository structure, service/frontend split, entry points, and runtime composition.
- [`architecture/service_boundaries.md`](architecture/service_boundaries.md)
  - Backend dependency direction, import constraints, and ownership rules.
- [`architecture/api_surface.md`](architecture/api_surface.md)
  - Core-service and ML-service route ownership.
- [`architecture/persistence_and_packages.md`](architecture/persistence_and_packages.md)
  - Shared backend workspace, persistence ownership, and validation expectations.

### Coding

- [`coding/overview.md`](coding/overview.md)
  - Index for code standards and quality expectations.
- [`coding/python.md`](coding/python.md)
  - Python runtime, typing, validation, async, structure, and service-boundary rules.
- [`coding/typescript.md`](coding/typescript.md)
  - Frontend TypeScript structure, contracts, API usage, and accessibility expectations.
- [`coding/quality_gates.md`](coding/quality_gates.md)
  - Linting, typing, testing, and validation requirements across the stack.
- [`coding/windows_scripts.md`](coding/windows_scripts.md)
  - Rules for `.bat` and PowerShell operational code.

### Runtime

- [`runtime/overview.md`](runtime/overview.md)
  - Index for runtime modes, startup, configuration, and packaging.
- [`runtime/modes.md`](runtime/modes.md)
  - Supported execution modes and their responsibilities.
- [`runtime/startup.md`](runtime/startup.md)
  - Launcher and manual startup procedures.
- [`runtime/configuration.md`](runtime/configuration.md)
  - Environment variables, structured settings, and mode-specific configuration behavior.
- [`runtime/deployment.md`](runtime/deployment.md)
  - Packaging, runtime dependencies, interoperability, and current constraints.

### UI

- [`ui/overview.md`](ui/overview.md)
  - Index for tokens, component patterns, and UX rules.
- [`ui/design_tokens.md`](ui/design_tokens.md)
  - Typography, spacing, radius, color, and surface standards.
- [`ui/components_and_patterns.md`](ui/components_and_patterns.md)
  - Buttons, forms, navigation, data views, dialogs, and status-heavy interaction patterns.
- [`ui/experience.md`](ui/experience.md)
  - Page structure, workflow UX, responsiveness, accessibility, and design principles.

### Operations

- [`operations/overview.md`](operations/overview.md)
  - Index for end-user workflows, common commands, and troubleshooting.
- [`operations/workflows.md`](operations/workflows.md)
  - Primary user workflows across source, fitting, and training.
- [`operations/commands.md`](operations/commands.md)
  - Launch, test, frontend, maintenance, and packaging commands.
- [`operations/troubleshooting.md`](operations/troubleshooting.md)
  - Common runtime and packaging issues with expected checks.

## Reading Order

1. Read this root index.
2. Open the relevant topic overview.
3. Open the smallest leaf file needed for the task.
4. Return here when switching to a different topic branch.

## Context Rules

- Read documentation only when required by the current task.
- Keep all affected documents aligned with implementation changes.
- Always include a `Last updated: YYYY-MM-DD` line when modifying a document.
- Do not read the entire documentation tree unless the task explicitly requires broad context.

## Environment Rules

- Assume Windows as the default operating environment.
- Document both CMD and PowerShell forms when commands differ.
- Prefer repository launch/build/test scripts over ad-hoc manual procedures.
- Update environment guidance when new runtime constraints or supported workflows are introduced.
