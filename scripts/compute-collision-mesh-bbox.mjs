/**
 * Computes axis-aligned bounds of AsteroidPsyche_Collision.glb vertices after the same
 * scale + offset Rust applies in rust_engine/src/lib.rs (load_collision_mesh).
 * Run: node scripts/compute-collision-mesh-bbox.mjs
 */
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const glbPath = path.join(__dirname, '..', 'public', 'models', 'AsteroidPsyche_Collision.glb');

const SCALE = 2.5;
const OFFSET = new THREE.Vector3(-3.75, -2.2, 3.22);

const buf = fs.readFileSync(glbPath);
const loader = new GLTFLoader();
const gltf = await loader.parseAsync(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength), '');

const box = new THREE.Box3();
const tmp = new THREE.Vector3();

gltf.scene.updateMatrixWorld(true);
gltf.scene.traverse((obj) => {
    if (!obj.isMesh || !obj.geometry) return;
    const pos = obj.geometry.getAttribute('position');
    if (!pos) return;
    const m = obj.matrixWorld;
    for (let i = 0; i < pos.count; i++) {
        tmp.fromBufferAttribute(pos, i).applyMatrix4(m);
        tmp.multiplyScalar(SCALE).add(OFFSET);
        box.expandByPoint(tmp);
    }
});

const size = new THREE.Vector3();
box.getSize(size);
const center = new THREE.Vector3();
box.getCenter(center);

console.log(JSON.stringify({
    min: box.min.toArray(),
    max: box.max.toArray(),
    size: size.toArray(),
    center: center.toArray(),
    scaleFactor: SCALE,
    offset: OFFSET.toArray(),
}, null, 2));
