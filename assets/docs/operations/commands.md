# ADSMOD Operational Commands

Last updated: 2026-06-03

## Launch And Maintenance

- Unified launcher and maintenance menu: `ADSMOD\start_on_windows.bat`

## Tests

- Full runner: `tests\run_tests.bat`
- Direct pytest: `.\app\server\.venv\Scripts\python.exe -m pytest app\tests -v`

## Frontend Development

- Core frontend dev server: run `npm run dev` from `ADSMOD\app\client`
- Frontend build: run `npm run build` from the relevant frontend directory

## Desktop Packaging

- Package desktop app: `release\tauri\build_with_tauri.bat`
