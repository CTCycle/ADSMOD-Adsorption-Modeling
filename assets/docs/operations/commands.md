# ADSMOD Operational Commands

Last updated: 2026-06-03

## Launch And Maintenance

- Start app: `ADSMOD\start_on_windows.bat`
- Maintenance menu: `ADSMOD\setup_and_maintenance.bat`

## Tests

- Full runner: `tests\run_tests.bat`
- Direct pytest: `.\app\server\.venv\Scripts\python.exe -m pytest app\tests -v`

## Frontend Development

- Core frontend dev server: run `npm run dev` from `ADSMOD\app\client`
- Frontend build: run `npm run build` from the relevant frontend directory

## Desktop Packaging

- Package desktop app: `release\tauri\build_with_tauri.bat`
