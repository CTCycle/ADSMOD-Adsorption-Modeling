import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';

const normalizeApiBaseUrl = (rawValue) => {
  const trimmed = String(rawValue || '').trim();
  if (!trimmed || /^https?:\/\//i.test(trimmed) || trimmed.startsWith('//')) {
    return '/api';
  }

  const withLeadingSlash = trimmed.startsWith('/') ? trimmed : `/${trimmed}`;
  return /^\/[A-Za-z0-9/_-]*$/.test(withLeadingSlash)
    ? withLeadingSlash.replace(/\/+$/, '') || '/api'
    : '/api';
};

const outputDir = path.resolve('src/assets');
await mkdir(outputDir, { recursive: true });
await writeFile(
  path.join(outputDir, 'runtime-config.json'),
  `${JSON.stringify({ apiBaseUrl: normalizeApiBaseUrl(process.env.VITE_API_BASE_URL) }, null, 2)}\n`
);
