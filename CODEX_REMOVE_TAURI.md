You are working on the ADSMOD-Adsorption-Modeling repository checked out at the current directory.

TASK: Remove all Tauri packaging infrastructure, consolidate the launcher scripts into a single PowerShell menu, and update all documentation. Do NOT modify any Python, TypeScript, or Angular source code.

## IMPORTANT: ADSMOD's Tauri project lives at `app/client/src-tauri/` (NOT `app/src-tauri/`)

## Step 1: Create `app.ps1` at repo root

Replace both `start_on_windows.bat` and `setup_and_maintenance.bat` with a single `app.ps1` interactive menu.

Menu title: "ADSMOD — Adsorption Modeling"

The menu options and logic are identical to PROMPT 1 Step 1. Read the existing batch files in this repo for exact paths, ports, and defaults.

## Step 2: Delete old batch files

- start_on_windows.bat
- setup_and_maintenance.bat

## Step 3: Delete all Tauri / Cargo / Rust artifacts

Directories to delete (entire trees):
- app/client/src-tauri/ (Cargo.toml, build.rs, capabilities/, gen/, icons/, src/, tauri.conf.json)
- app/client/scripts/clean-tauri-icons.ps1
- release/tauri/
- release/windows/ (if exists)

Note: There is NO .github/workflows/desktop-release.yml in this repo — only ci.yml exists. Leave ci.yml unchanged.

## Step 4: Update .gitignore

Remove entries for Tauri build outputs (note paths use app/client/src-tauri/ rather than app/src-tauri/).

## Step 5: Update package.json (app/client/)

- Remove "@tauri-apps/cli" from devDependencies if present
- Remove any "build:tauri" script

## Step 6: Update README.md

Read the current README.md and make these changes:
- Remove the "Windows (Packaged Tauri App):" build reference
- Remove "release\tauri\build_with_tauri.bat" and "release/windows/installers|portable" mentions
- Remove the "Clean desktop build artifacts" maintenance option reference from the setup menu list
- Simplify runtime description to only cover local web mode
- Remove any `.env` keys reference for "Tauri startup"
- Update batch file references to app.ps1

## Step 7: Update assets/docs/

Scan for Tauri/desktop references.

## Step 8: Verify

Check: app/client/src-tauri/ gone, release/tauri/ gone, app.ps1 exists.