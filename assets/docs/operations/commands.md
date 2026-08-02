# ADSMOD Operational Commands

Last updated: 2026-08-02

## Launch And Maintenance

- Unified launcher and maintenance menu: `powershell -ExecutionPolicy Bypass -File .\start_on_windows.ps1`

## Tests

- Full runner: `app\tests\run_tests.bat`
- Direct pytest: `.\app\server\.venv\Scripts\python.exe -m pytest app\tests -v`
- SQLite persistence contract: `.\app\server\.venv\Scripts\python.exe -m pytest app\tests\persistence -v`

## Frontend Development

- Unified frontend dev server: run `npm run dev` from `app\client`
- Frontend build: run `npm run build` from `app\client`
- Frontend visual comparison: run `npm run visual:compare` from `app\client`
