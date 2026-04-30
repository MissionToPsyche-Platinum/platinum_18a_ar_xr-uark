/**
 * Rebases AR.js area config from marker-0 frame to marker-4 frame:
 * M'_i = inv(M_4) * M_i  → marker 4 becomes identity; others relative to 4.
 * Usage: node scripts/rebase-config-to-marker4.mjs [path/to/config.json]
 */
import { readFileSync, writeFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { Matrix4 } from 'three';

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = join(__dirname, '..');
const target = process.argv[2] || join(root, 'public', 'config.json');

const raw = readFileSync(target, 'utf8');
const json = JSON.parse(raw);
const controls = json.subMarkersControls;
if (!Array.isArray(controls)) {
    console.error('Invalid config: missing subMarkersControls');
    process.exit(1);
}

const m4Entry = controls.find((p) => p.parameters?.barcodeValue === 4);
if (!m4Entry?.poseMatrix?.length) {
    console.error('Invalid config: marker 4 pose missing');
    process.exit(1);
}

const inv4 = new Matrix4().fromArray(m4Entry.poseMatrix).invert();

for (const item of controls) {
    if (!item.poseMatrix?.length) continue;
    const m = new Matrix4().fromArray(item.poseMatrix);
    const out = inv4.clone().multiply(m);
    item.poseMatrix = Array.from(out.elements);
}

writeFileSync(target, JSON.stringify(json, null, 2) + '\n', 'utf8');
console.log('Rebased to marker-4 frame (marker 4 = identity):', target);
