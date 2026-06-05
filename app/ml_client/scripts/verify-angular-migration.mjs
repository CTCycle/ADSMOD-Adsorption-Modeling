import { readdir, readFile } from 'node:fs/promises';
import { dirname, extname, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const srcRoot = join(__dirname, '..', 'src');

const skippedDirectories = new Set([
  '.angular',
  'coverage',
  'dist',
  'node_modules',
  'out-tsc',
]);

const forbiddenExtensions = new Set(['.jsx', '.tsx']);
const scannedTextExtensions = new Set(['.cjs', '.css', '.html', '.js', '.mjs', '.ts']);

const forbiddenPatterns = [
  { label: 'React import', pattern: /from\s+['"]react['"]/u },
  { label: 'ReactDOM import', pattern: /from\s+['"]react-dom(?:\/client)?['"]/u },
  { label: 'ReactDOM usage', pattern: /\bReactDOM\b/u },
  { label: 'React createRoot usage', pattern: /\bcreateRoot\s*\(/u },
  { label: 'React hook usage', pattern: /\buse(State|Effect|Memo|Callback|Ref|Reducer)\s*\(/u },
  { label: 'TSX import path', pattern: /\.tsx['"]/u },
];

const violations = [];

async function walk(directory) {
  const entries = await readdir(directory, { withFileTypes: true });

  for (const entry of entries) {
    const absolutePath = join(directory, entry.name);
    const relativePath = relative(srcRoot, absolutePath).replaceAll('\\', '/');

    if (entry.isDirectory()) {
      if (!skippedDirectories.has(entry.name)) {
        await walk(absolutePath);
      }
      continue;
    }

    if (!entry.isFile()) {
      continue;
    }

    const extension = extname(entry.name).toLowerCase();

    if (forbiddenExtensions.has(extension)) {
      violations.push(`${relativePath}: forbidden React-era file extension "${extension}"`);
    }

    if (!scannedTextExtensions.has(extension)) {
      continue;
    }

    const content = await readFile(absolutePath, 'utf8');
    for (const { label, pattern } of forbiddenPatterns) {
      if (pattern.test(content)) {
        violations.push(`${relativePath}: ${label}`);
      }
    }
  }
}

await walk(srcRoot);

if (violations.length > 0) {
  console.error('Angular migration verification failed:');
  for (const violation of violations) {
    console.error(`- ${violation}`);
  }
  process.exitCode = 1;
} else {
  console.log('Angular migration verification passed.');
}