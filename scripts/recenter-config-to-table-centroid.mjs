/**
 * Recenter AR area config translations to the centroid of table markers (1,3,4,6).
 * Usage:
 *   node scripts/recenter-config-to-table-centroid.mjs [path/to/config.json]
 */
import { readFileSync, writeFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = join(__dirname, '..');
const target = process.argv[2] || join(root, 'public', 'config.json');

const TABLE_IDS = new Set([1, 3, 4, 6]);

const raw = readFileSync(target, 'utf8');
const json = JSON.parse(raw);
const controls = json.subMarkersControls;
if (!Array.isArray(controls)) {
  console.error('Invalid config: missing subMarkersControls');
  process.exit(1);
}

const rows = controls
  .map((row) => ({
    id: row?.parameters?.barcodeValue,
    pose: row?.poseMatrix,
    row,
  }))
  .filter((r) => typeof r.id === 'number' && Array.isArray(r.pose) && r.pose.length >= 16);

const table = rows.filter((r) => TABLE_IDS.has(r.id));
if (table.length < 3) {
  console.error('Need at least 3 table markers from [1,3,4,6] to compute centroid.');
  process.exit(1);
}

let sx = 0;
let sy = 0;
let sz = 0;
for (const r of table) {
  sx += Number(r.pose[12]);
  sy += Number(r.pose[13]);
  sz += Number(r.pose[14]);
}
const cx = sx / table.length;
const cy = sy / table.length;
const cz = sz / table.length;

for (const r of rows) {
  r.pose[12] = Number(r.pose[12]) - cx;
  r.pose[13] = Number(r.pose[13]) - cy;
  r.pose[14] = Number(r.pose[14]) - cz;
}

writeFileSync(target, JSON.stringify(json, null, 2) + '\n', 'utf8');
console.log(
  `Recentered config to table centroid (${cx.toFixed(6)}, ${cy.toFixed(6)}, ${cz.toFixed(6)}): ${target}`
);
