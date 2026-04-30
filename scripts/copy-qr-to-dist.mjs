/**
 * Copies Vite project `qr/` (qr.html + qrcode.min.js) → `dist/qr/` for GitHub Pages
 * (e.g. https://<user>.github.io/nasa-psyche-ar/qr/qr.html).
 *
 * Edit `nasa-psyche-ar/qr/` only (same folder as `src/`). `dist/qr/` is overwritten every build.
 */
import { cpSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const viteRoot = join(__dirname, '..');
const qrSource = join(viteRoot, 'qr');
const qrDest = join(viteRoot, 'dist', 'qr');

if (!existsSync(join(viteRoot, 'dist'))) {
  console.warn('[copy-qr] dist/ missing — run vite build first.');
  process.exit(1);
}
if (!existsSync(qrSource)) {
  console.warn('[copy-qr] Skipping: not found:', qrSource);
  process.exit(0);
}

cpSync(qrSource, qrDest, { recursive: true });
console.log('[copy-qr] Copied', qrSource, '→', qrDest);
