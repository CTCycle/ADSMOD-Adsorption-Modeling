# README Writing Guidelines

Use this template to write or update README files in this repository.

Goal: produce user-facing documentation that is accurate, concise, and aligned with current project behavior.

## Required Structure

Keep this section order. Omit sections that do not apply, then renumber.

## 1. Project Overview

- Describe what ADSMOD does and who it is for.
- Explain high-level backend/frontend interaction without code internals.
- Avoid implementation details (classes/functions/module internals).

## 2. Model and Dataset (Optional)

Use for ML-related README sections only.

- Describe model category at a high level.
- Describe dataset origin and nature (uploaded, generated, external).
- State uncertainty explicitly when details are unknown.

## 3. Installation

Provide minimal reproducible setup instructions.

### 3.1 Windows

- Prefer launcher-based setup (`ADSMOD\start_on_windows.bat`) when applicable.
- Explain first-run vs subsequent-run behavior at a high level.
- Clarify portability/local side effects.

### 3.2 macOS / Linux (if supported)

- List prerequisites and manual steps.
- Separate backend/frontend startup if needed.

## 4. How to Use

Document operational workflow:
- load data,
- run fitting/training tasks,
- inspect outputs and status.

For UI workflows, include screenshots when available from project assets.

## 5. Setup and Maintenance

List operational scripts/utilities and what they do (not internals), such as:
- test runner,
- environment bootstrap,
- cleanup/reset actions.

## 6. Resources

Document `ADSMOD/resources` usage:
- what each relevant subfolder stores,
- how runtime uses it.

## 7. Configuration

Describe active configuration files:
- `ADSMOD/settings/.env`,
- template profiles in `ADSMOD/settings/`,
- any related configuration JSON used by the app.

Include a variable table:

| Variable | Description |
|---|---|
| `VARIABLE_NAME` | Purpose, where defined, and default behavior/value |

## 8. License

State the license and point to `LICENSE`.

## Writing Rules

- Write for users/operators first.
- Keep statements factual and verifiable from the current repository.
- Do not document stale endpoints, scripts, or unsupported modes.
- Keep language skimmable with short paragraphs and focused lists.
