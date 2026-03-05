# ADSMOD Frontend

This is the React + TypeScript frontend for ADSMOD.

## Development

Use your machine local Node.js/npm runtime for all npm commands.

Example (PowerShell, no profile):
```powershell
& "C:\Program Files\PowerShell\7\pwsh.exe" -NoProfile -Command "cmd /c npm run build"
```

Install dependencies:
```bash
npm install
```

Run development server:
```bash
npm run dev
```

Build for production:
```bash
npm run build
```

## Configuration

The frontend connects to the FastAPI backend through the API base path configured via `VITE_API_BASE_URL` (default: `/api`).

Runtime host/port and API proxy values are resolved from `ADSMOD/settings/.env` via `vite.config.ts`.

## Technology Stack

- **React 18** - UI framework
- **TypeScript** - Type-safe development
- **Vite** - Fast build tool and dev server
- **CSS3** - Modern styling with custom design system
