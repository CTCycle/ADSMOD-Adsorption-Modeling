# ADSMOD Operational Commands

Last updated: 2026-07-11

## Launch And Maintenance

- Unified launcher and maintenance menu: `powershell -ExecutionPolicy Bypass -File ADSMOD\start_on_windows.ps1`

## Tests

- Full runner: `tests\run_tests.bat`
- Direct pytest: `.\app\server\.venv\Scripts\python.exe -m pytest app\tests -v`

## Frontend Development

- Unified frontend dev server: run `npm run dev` from `ADSMOD\app\client`
- Frontend build: run `npm run build` from `ADSMOD\app\client`
