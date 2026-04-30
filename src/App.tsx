/**
 * NASA Psyche AR — Web/AR rover exploration experience.
 * Uses React + A-Frame for 3D, Rust/WASM for collision and movement.
 */
import { useEffect, useLayoutEffect, useState, useCallback, useRef } from 'react';
import MODE_CONFIG, { Difficulty } from './modeConfig';
import { augmentHeightmapWithDepthFromMotion } from './arDepthFromMotion';
// @ts-ignore
import init, { start_ar_session, load_collision_mesh, move_rover_on_asteroid, get_surface_point_in_direction } from '../rust_engine/pkg/rust_engine';

/** Returns a uniformly distributed random unit vector on the sphere via rejection sampling. */
const randomUnitVector = (): [number, number, number] => {
    let x: number, y: number, z: number, len: number;
    do {
        x = Math.random() * 2 - 1;
        y = Math.random() * 2 - 1;
        z = Math.random() * 2 - 1;
        len = Math.sqrt(x * x + y * y + z * z);
    } while (len === 0 || len > 1);
    return [x / len, y / len, z / len];
};

const MOVE_INTERVAL = 33; // ms between movement ticks (~30 fps)
const MAX_ENERGY = 50;
/** AR rover moves at a smaller step size than web since the marker-anchored world is smaller.
 *  Tuned down so the rover crawls across the captured terrain and the physics has time to
 *  sample the heightmap faithfully at each tick. */
const AR_ROVER_SPEED_SCALE = 0.009375;
/**
 * Flat AR: keep rover, samples, and obstacles inside this fraction of the mapped disk radius.
 * Tighter than the full tap radius so content stays over the reliable gray snapshot / height field.
 */
const AR_FLAT_PLAY_INNER_RADIUS_FR = 0.72;
/**
 * Flat AR: subtract this from heightmap-based Y so props read grounded on the print instead of
 * hovering slightly above the displaced mesh.
 */
const AR_FLAT_SURFACE_Y_SINK = 0.022;
/** Flat AR: rover GLB + collection / arrow sizing vs the prior baseline (same craft as web). */
const AR_ROVER_SCALE_MULTIPLIER = 5;
/**
 * Flat AR: place samples and obstacles within this physical radius (meters) of the rover
 * (disk center on the scanned surface). 10 in ≈ 0.254 m; divided by MARKER_SIZE_METERS for marker units.
 */
const AR_FLAT_SPAWN_CLUSTER_RADIUS_M = 10 * 0.0254;
/** Deterministic initial rover spawn direction in asteroid-local space for AR. */
const AR_ROVER_START_DIRECTION: [number, number, number] = [0, 1, 0];

/** Pushes a point radially outward from the asteroid center by `offset` so the rover hugs the surface without clipping. */
const pushOutFromCenter = (x: number, y: number, z: number, offset: number): [number, number, number] => {
    const len = Math.hypot(x, y, z);
    if (len < 1e-6) return [x, y, z];
    return [
        x + (x / len) * offset,
        y + (y / len) * offset,
        z + (z / len) * offset,
    ];
};

/**
 * AR alignment tunables — live-editable through the AR calibration panel and persisted in
 * localStorage. Defaults reflect the values baked into the prototype; use the in-AR sliders
 * to drive these until the virtual asteroid matches the physical asteroid, then copy the
 * resulting JSON back into AR_CALIBRATION_DEFAULTS to ship them.
 */
type ArCalibration = {
    modelLift: number;
    modelBack: number;
    modelYawOffsetDeg: number;
    modelPitchOffsetDeg: number;
    modelRollOffsetDeg: number;
    modelScaleX: number;
    modelScaleY: number;
    modelScaleZ: number;
    sampleScaleFr: number;
    /** When true, changing Lift (Y) multiplies X/Y/Z scale so apparent size stays ~constant (deeper → bigger). */
    compensateScaleWithLift: boolean;
    /** Depth proxy = pivot − lift (marker units); default 0 matches “more negative Y = farther → scale up”. */
    liftDistancePivot: number;
    /**
     * Hide the virtual asteroid GLB in AR while keeping it alive as physics surface.
     * User then sees the real 3D-printed asteroid through the camera with rover/crystals/obstacles
     * riding the (invisible) digital twin. Toggle OFF during calibration to see virtual vs physical drift.
     */
    hideVirtualAsteroid: boolean;
    /** Red 2"×2" reference cube on every marker — visible check for AR.js tracking & calibration math. */
    showCalibrationCube: boolean;
    /**
     * When on (recommended for the printed prop), the rover walks on a flat horizontal disk anchored
     * to the active marker instead of the full 3D asteroid surface. This matches the user's view — the
     * physical print shows one face, so wrapping around a virtual sphere is invisible and confusing.
     * Samples + obstacles are redistributed on the same disk. Disable to fall back to full-mesh physics.
     */
    flatSurfaceMode: boolean;
    /** Flat disk radius in scaled-local units (inside the scale transform; tap-to-setup writes this). */
    flatSurfaceRadius: number;
    /** Disk height above the marker plane (scaled-local units). */
    flatSurfaceHeight: number;
    /** Disk center offset X in scaled-local units (set by the user's first tap during setup). */
    flatSurfaceOffsetX: number;
    /** Disk center offset Z in scaled-local units (set by the user's first tap during setup). */
    flatSurfaceOffsetZ: number;
    /** Visualize the flat disk (semi-transparent green ring). */
    showFlatDisk: boolean;
    /**
     * **Visual only.** When true, the displaced snapshot mesh (`heightmap-terrain`) is not drawn,
     * so you see the real print through the camera while the rover/samples still use the same
     * heightmap (`terrainRef`, `sampleHeightmap`, `stepOnHeightmap`, etc.) — no change to math.
     */
    hideTerrainSurface: boolean;
};

/** Generates star data with uniform random distribution across a surrounding sphere. */
const generateStars = (count: number) => {
    const COLORS = ['#FFFFFF', '#FFFFFF', '#FFFFFF', '#FFFFFF', '#00d4ff', '#7b2cbf'];
    const RADIUS = 120;
    // Simple seeded LCG for fully deterministic placement and variation
    let seed = 42;
    const rand = () => { seed = (seed * 1664525 + 1013904223) & 0xffffffff; return (seed >>> 0) / 0xffffffff; };
    return Array.from({ length: count }, (_, i) => {
        const phi = Math.acos(2 * rand() - 1);       // uniform latitude (0..π)
        const theta = 2 * Math.PI * rand();             // uniform longitude (0..2π)
        const radius = RADIUS + (rand() - 0.5) * 24;    // ±12 units of depth jitter
        const x = Math.sin(phi) * Math.cos(theta) * radius;
        const yPos = Math.cos(phi) * radius;
        const z = Math.sin(phi) * Math.sin(theta) * radius;
        return {
            id: i,
            pos: `${x.toFixed(2)} ${yPos.toFixed(2)} ${z.toFixed(2)}`,
            radius: 0.3 + rand() * 0.4,
            color: COLORS[Math.floor(rand() * COLORS.length)],
            opacity: 0.7 + rand() * 0.3,
            dur: Math.round(2000 + rand() * 3000),
            delay: Math.round(rand() * 2000),
        };
    });
};

const STARS = generateStars(250);

const INTRO_CONTENT: Record<string, { welcome: string; description: string }> = {
    easy: {
        welcome: 'Welcome to Story Mode',
        description: 'Explore the surface of asteroid Psyche with complete freedom. Pilot the rover across the terrain and drive over samples to collect them. If you ever get lost, follow the indicator arrow to the nearest sample.',
    },
    normal: {
        welcome: 'Welcome to Standard Mode',
        description: "Explore Psyche with the energy system in play. Your rover's battery drains as you roam — collect samples efficiently before power runs out. Follow the indicator arrow if you lose track of your next sample. The mission ends when you collect all 20 samples or run out of energy.",
    },
    hard: {
        welcome: 'Welcome to Challenge Mode',
        description: "Psyche is at its most unforgiving. Energy drains your battery, and craters larger than the rover are scattered across the surface — driving into one cuts your speed in half and drains energy faster. Navigate carefully, collect samples quickly, and use the indicator arrow wisely. The mission ends when you collect all 20 samples or run out of energy.",
    },
};
const OBSTACLE_DIRECTIONS: [number, number, number, number][] = [
    [0.6849, 2.2127, -1.16, 1.15],
    [-0.6158, 2.7743, 0.4824, .8],
    [1.68, 0.35, 2.861, .26],
    [2.9, 1.65, 0.633, 0.27],
    [3.3426, -0.4972, 0.08, 0.35],
    [1.9917, 2.33, 1.4612, 0.24],
    [-3.5883, -0.1327, -.01, .35],
    [-0.4975, -0.5, 2.7357, 0.24],
    [-1.21, -1.4, 1.6656, 0.35],
    [-2.5683, -1.06, 1.3699, 0.2],
    [-2.5431, -1.3294, 0.7099, 0.25],
    [-0.13, -1.88, -1.1987, 0.15],
    [-1.5, -0.59, 2.498, 0.21],
    [1.72, -0.72, 2.4154, 0.22],
    [1.05, 2.58, 2.1314, 0.15],
    [2.9039 , 0.4 , -2.0303, 0.25],
    [0.66 , -1.76 , 1.8281 , 0.15],
    [-1.13 , -0.92 , 2.3459 , 0.15],
    [-2.4586 , 1.45 , -1.8348 , 0.24],
    [0.835 , -0.66 , -2.8635 , 0.28],
    [2.8161 , -1.215 , -0.1877 , 0.19],
    [2.6203 , -1.15 , 0.5583 , 0.18],
    [2.8735 , 0.14 , 1.709 , 0.15],
    [-1.9249 , 2.27 , 1.6324 , 0.17],
    [-2.9 , 1.88 , -0.514, 0.23],
    [-1.335 , 1.97 , 2.3023 , 0.17],
    [2.108 , -1.55 , -1.0963 , 0.17],
    [2.9755 , 1.9092 , -1.0774 , 0.17],
];

type SampleModel = 'crystal' | 'ore' | 'rock';

/** ------------------------------------------------------------------
 * AR calibration types & helpers (marker pose-based anchoring).
 * Pulled from the calibrated AR prototype to keep scale/orientation
 * consistent between table and surface markers.
 * ------------------------------------------------------------------ */
type MarkerOffset = { x: number; y: number; z: number };
type MarkerPoseSource = { byId: Record<number, number[]>; center?: MarkerOffset };
const TABLE_MARKER_IDS = [1, 3, 4, 6] as const;
const SURFACE_MARKER_IDS = [0, 2, 5, 7] as const;
const ALL_MARKER_IDS = [...TABLE_MARKER_IDS, ...SURFACE_MARKER_IDS] as const;
const MARKER_SIZE_METERS = 0.0508; // 2 inches printed barcode

/**
 * 3D-printed physical asteroid (meters), same axis convention as public/surface_pair_report.json
 * ("three.js Y-up; table flattened to Y=0; surface markers constrained to +Y").
 * Mesh spans: npm run ar:collision-bbox (scripts/compute-collision-mesh-bbox.mjs).
 */
const PHYSICAL_ASTEROID_BBOX_M = { x: 0.61, y: 0.524, z: 0.432 } as const;
/** AABB edge lengths of AsteroidPsyche_Collision.glb after Rust scale 2.5 + offset (physics space). */
const COLLISION_MESH_PHYSICS_SPAN = { x: 7.407302185893059, y: 5.180009913165122, z: 6.379854867700487 } as const;
/** Legacy mean scale (7.2/6/6) — only used to rescale lift when applying physical match. */
const LEGACY_AR_MODEL_SCALE_REF = (7.2 + 6.0 + 6.0) / 3;

/** Uniform modelScale (X=Y=Z) so WASM stays isotropic; least-squares fit of bbox edges to meters. */
function computeUniformScaleForPhysicalAsteroid(): number {
    const sx = COLLISION_MESH_PHYSICS_SPAN.x;
    const sy = COLLISION_MESH_PHYSICS_SPAN.y;
    const sz = COLLISION_MESH_PHYSICS_SPAN.z;
    const px = PHYSICAL_ASTEROID_BBOX_M.x;
    const py = PHYSICAL_ASTEROID_BBOX_M.y;
    const pz = PHYSICAL_ASTEROID_BBOX_M.z;
    const dot = sx * px + sy * py + sz * pz;
    const normSq = sx * sx + sy * sy + sz * sz;
    return dot / (MARKER_SIZE_METERS * normSq);
}

const AR_PHYSICAL_MATCH_UNIFORM_SCALE = computeUniformScaleForPhysicalAsteroid();

const AR_CALIBRATION_DEFAULTS: ArCalibration = {
    modelLift: (-30.15 * AR_PHYSICAL_MATCH_UNIFORM_SCALE) / LEGACY_AR_MODEL_SCALE_REF,
    modelBack: 0.0,
    modelYawOffsetDeg: 180,
    modelPitchOffsetDeg: 35,
    modelRollOffsetDeg: 0,
    modelScaleX: AR_PHYSICAL_MATCH_UNIFORM_SCALE,
    modelScaleY: AR_PHYSICAL_MATCH_UNIFORM_SCALE,
    modelScaleZ: AR_PHYSICAL_MATCH_UNIFORM_SCALE,
    sampleScaleFr: 0.20,
    compensateScaleWithLift: false,
    liftDistancePivot: 0,
    // "Digital twin" mode on by default: real asteroid visible through camera, rover/crystals on invisible twin.
    hideVirtualAsteroid: true,
    showCalibrationCube: false,
    // Flat-surface mode by default — avoids wrap-around-sphere issue when only one face is visible.
    flatSurfaceMode: true,
    // Placeholder radius; the user's tap-to-map flow overwrites this before game start.
    flatSurfaceRadius: (PHYSICAL_ASTEROID_BBOX_M.x / 2) / MARKER_SIZE_METERS,
    flatSurfaceHeight: 0,
    flatSurfaceOffsetX: 0,
    flatSurfaceOffsetZ: 0,
    showFlatDisk: false,
    /** Start with overlay off so motion reads against the physical rock (toggle on to see the mesh). */
    hideTerrainSurface: true,
};

// v7: hideTerrainSurface (visual-only mesh; heightmap math unchanged). Default true for new installs.
const AR_CALIBRATION_STORAGE_KEY = 'nasa-psyche-ar-calibration-v7';

const loadArCalibration = (): ArCalibration => {
    try {
        const raw = localStorage.getItem(AR_CALIBRATION_STORAGE_KEY);
        if (!raw) return { ...AR_CALIBRATION_DEFAULTS };
        const parsed = JSON.parse(raw) as Partial<ArCalibration>;
        return { ...AR_CALIBRATION_DEFAULTS, ...parsed };
    } catch {
        return { ...AR_CALIBRATION_DEFAULTS };
    }
};

/** Positive “depth” proxy for lift compensation; pivot − lift, floored so we never divide by ~0. */
function liftDepthForScreenCompensation(pivot: number, lift: number): number {
    return Math.max(0.25, pivot - lift);
}

/**
 * Planar rover step for flat-surface mode: walk on a horizontal disk of `radius` centered at
 * (centerX, height, centerZ) in the asteroid-parent frame. No raycasting, no wrap-around —
 * rover stays on the visible face of the physical print. Returns clamped position.
 */
function stepOnFlatDisk(
    current: { x: number; y: number; z: number },
    moveX: number,
    moveZ: number,
    radius: number,
    height: number,
    centerX: number,
    centerZ: number,
): { x: number; y: number; z: number } {
    const newX = current.x + moveX;
    const newZ = current.z + moveZ;
    const dx = newX - centerX;
    const dz = newZ - centerZ;
    const r = Math.hypot(dx, dz);
    const clampR = radius * AR_FLAT_PLAY_INNER_RADIUS_FR;
    const y = height - AR_FLAT_SURFACE_Y_SINK;
    if (r <= clampR || r < 1e-6) {
        return { x: newX, y, z: newZ };
    }
    const s = clampR / r;
    return { x: centerX + dx * s, y, z: centerZ + dz * s };
}

/** Uniform-area sample within a disk — sqrt on the radius gives even distribution. */
function randomPointOnDisk(radius: number): { x: number; z: number } {
    const theta = Math.random() * Math.PI * 2;
    const r = Math.sqrt(Math.random()) * radius;
    return { x: Math.cos(theta) * r, z: Math.sin(theta) * r };
}

/**
 * A photographed terrain: once the player finishes tap-to-map setup we grab a frame, crop the
 * circular play zone, and convert it into a luminance-derived heightmap. Values are normalized
 * so roughly half of the range is above and half below the disk plane; `scaleY` scales that
 * range into scaled-local units (same coordinate frame the rover + samples live in).
 */
export type TerrainHeightmap = {
    data: Float32Array; // length = size*size, values in [0, 1] (0.5 = average height)
    size: number;       // grid side (e.g. 96)
    scaleY: number;     // world-space amplitude of height deviation around the disk plane
    /**
     * Peak height of the paraboloidal base dome — added to every sample so the terrain
     * **wraps** around the physical asteroid instead of sitting flat on the marker. 0 means no
     * dome (flat base + detail on top), larger values push the center of the disk upward.
     */
    domeHeight: number;
    /**
     * Exponent used to curve the detail heightmap. >1 exaggerates peaks; <1 softens them. 1.6
     * gives a noticeably "wrinkled" surface without turning bumps into spikes.
     */
    detailGamma: number;
    textureUrl: string; // data URL of the cropped snapshot (used to texture the terrain mesh)
    sourceFrameW: number;
    sourceFrameH: number;
    centerPx: { x: number; y: number }; // circle center in source frame pixels
    radiusPx: number;                    // circle radius in source frame pixels
    /** Video-pixel crop used for OpenCV (same rect as texture crop) — feeds depth-from-motion. */
    cropRect: { x: number; y: number; size: number };
    /** Jet heatmap of the motion-depth proxy (see `augmentHeightmapWithDepthFromMotion`). */
    depthDebugUrl?: string;
};

/**
 * Bilinearly sample the heightmap at disk-local coordinates. `x` and `z` are offsets from the
 * disk's center in world units; `radius` is the disk radius in world units. Returns a height in
 * world units (scaleY baked in) — safe to feed directly into a Y position.
 */
function sampleHeightmap(
    hm: TerrainHeightmap,
    x: number,
    z: number,
    radius: number,
): number {
    if (radius <= 0) return 0;
    // Radial falloff (1 at center, 0 at rim) — used both to taper the dome and to fade the
    // detail at the edges so the terrain dissolves cleanly into the marker plane.
    const rho = Math.hypot(x, z) / radius;
    const rhoClamped = Math.min(1, Math.max(0, rho));
    // Spherical-cap profile √(1−ρ²): lifts the mid-disk more than a parabola so the play surface
    // reads as a shallow dome “wrapping” the physical asteroid instead of a flat plate.
    const domeFalloff = Math.sqrt(Math.max(0, 1 - rhoClamped * rhoClamped));
    const detailFalloff = 1 - rhoClamped * rhoClamped;

    // Bilinear sample of the normalized pixel luminance.
    const u = (x / (2 * radius)) + 0.5;
    const v = (z / (2 * radius)) + 0.5;
    let pixel = 0.5;
    if (u >= 0 && u <= 1 && v >= 0 && v <= 1) {
        const s = hm.size;
        const fx = u * (s - 1);
        const fy = v * (s - 1);
        const x0 = Math.floor(fx);
        const y0 = Math.floor(fy);
        const x1 = Math.min(x0 + 1, s - 1);
        const y1 = Math.min(y0 + 1, s - 1);
        const tx = fx - x0;
        const ty = fy - y0;
        const h00 = hm.data[y0 * s + x0];
        const h10 = hm.data[y0 * s + x1];
        const h01 = hm.data[y1 * s + x0];
        const h11 = hm.data[y1 * s + x1];
        const h0 = h00 * (1 - tx) + h10 * tx;
        const h1 = h01 * (1 - tx) + h11 * tx;
        pixel = h0 * (1 - ty) + h1 * ty;
    }

    // Signed detail in [-1, +1] with a gamma curve for punchier peaks/valleys.
    const centered = (pixel - 0.5) * 2;
    const mag = Math.pow(Math.abs(centered), hm.detailGamma);
    const detail = Math.sign(centered) * mag;

    const dome = hm.domeHeight * domeFalloff;
    return dome + detail * hm.scaleY * detailFalloff;
}

/**
 * Central-difference gradient of the heightmap at (x, z). Returned normal is in world units and
 * already normalized. Used to tilt the rover to match the terrain slope.
 */
function sampleHeightmapNormal(
    hm: TerrainHeightmap,
    x: number,
    z: number,
    radius: number,
): { nx: number; ny: number; nz: number } {
    const eps = Math.max(radius * 0.02, 0.02);
    const hxp = sampleHeightmap(hm, x + eps, z, radius);
    const hxm = sampleHeightmap(hm, x - eps, z, radius);
    const hzp = sampleHeightmap(hm, x, z + eps, radius);
    const hzm = sampleHeightmap(hm, x, z - eps, radius);
    const dhdx = (hxp - hxm) / (2 * eps);
    const dhdz = (hzp - hzm) / (2 * eps);
    // Normal of surface y = h(x, z) is (-dh/dx, 1, -dh/dz), then normalize.
    const nx = -dhdx;
    const ny = 1;
    const nz = -dhdz;
    const len = Math.hypot(nx, ny, nz) || 1;
    return { nx: nx / len, ny: ny / len, nz: nz / len };
}

/**
 * Rover step on a heightmap-terrain disk: clamp XZ to the disk, then look up Y from the
 * heightmap. The returned position is snapped to the real surface so the rover bobs over
 * bumps and dips into valleys as it drives.
 */
function stepOnHeightmap(
    current: { x: number; y: number; z: number },
    moveX: number,
    moveZ: number,
    radius: number,
    diskY: number,
    centerX: number,
    centerZ: number,
    hm: TerrainHeightmap | null,
): { x: number; y: number; z: number } {
    const newX = current.x + moveX;
    const newZ = current.z + moveZ;
    const dx = newX - centerX;
    const dz = newZ - centerZ;
    const r = Math.hypot(dx, dz);
    const clampR = radius * AR_FLAT_PLAY_INNER_RADIUS_FR;
    let px: number;
    let pz: number;
    if (r <= clampR || r < 1e-6) {
        px = newX;
        pz = newZ;
    } else {
        const s = clampR / r;
        px = centerX + dx * s;
        pz = centerZ + dz * s;
    }
    const py =
        diskY
        + (hm ? sampleHeightmap(hm, px - centerX, pz - centerZ, radius) : 0)
        - AR_FLAT_SURFACE_Y_SINK;
    return { x: px, y: py, z: pz };
}

/**
 * AR flat-disk movement: map joystick (screen-style +X right, +Y up) onto the play disk in
 * `ar-play-root` local XZ. Uses the live camera orientation so "push up" moves toward the top
 * of the device view on the table — unlike `getCameraFrame` + `.xz`, which used a spherical
 * tangent at the rover and often inverted or crushed vertical input.
 */
function computeArFlatDiskMoveXZ(
    THREE: any,
    inputX: number,
    inputY: number,
    speed: number,
): { mx: number; mz: number } {
    const sceneEl = document.querySelector('a-scene') as any;
    const camObj = sceneEl?.camera?.el?.object3D;
    const playRoot = document.getElementById('ar-play-root') as any;
    if (!camObj || !playRoot?.object3D) {
        return { mx: inputX * speed, mz: inputY * speed };
    }

    playRoot.object3D.updateWorldMatrix(true, false);
    camObj.updateWorldMatrix(true, false);

    const playQuat = new THREE.Quaternion();
    playRoot.object3D.getWorldQuaternion(playQuat);
    const groundUp = new THREE.Vector3(0, 1, 0).applyQuaternion(playQuat).normalize();

    const camQuat = new THREE.Quaternion();
    camObj.getWorldQuaternion(camQuat);
    const camRight = new THREE.Vector3(1, 0, 0).applyQuaternion(camQuat);
    const camScrUp = new THREE.Vector3(0, 1, 0).applyQuaternion(camQuat);

    const proj = (v: any) => {
        const o = v.clone();
        o.addScaledVector(groundUp, -o.dot(groundUp));
        return o;
    };

    let r = proj(camRight);
    let u = proj(camScrUp);

    if (r.lengthSq() < 1e-10) {
        r = new THREE.Vector3(1, 0, 0);
        r.addScaledVector(groundUp, -r.dot(groundUp));
        if (r.lengthSq() < 1e-10) r.set(0, 0, 1);
        r.normalize();
    } else {
        r.normalize();
    }
    if (u.lengthSq() < 1e-10) {
        u = new THREE.Vector3().crossVectors(groundUp, r).normalize();
    } else {
        u.normalize();
        u.addScaledVector(r, -u.dot(r));
        if (u.lengthSq() < 1e-10) u = new THREE.Vector3().crossVectors(groundUp, r).normalize();
        else u.normalize();
    }

    const moveWorld = r.clone().multiplyScalar(inputX).addScaledVector(u, inputY).multiplyScalar(speed);
    const invPlayQuat = playQuat.clone().invert();
    moveWorld.applyQuaternion(invPlayQuat);
    return { mx: moveWorld.x, mz: moveWorld.z };
}

/**
 * OpenCV pipeline: take a video element + a circular crop in pixel space and return a
 * luminance-based heightmap. Bright pixels → peaks, dark pixels → valleys. We use CLAHE to
 * boost contrast of the natural lighting on the print (bumps get highlights, dents cast
 * shadows), then Gaussian-blur to smooth noise before downsampling to the grid resolution.
 *
 * Also returns a data URL of the cropped region — perfect for texturing the displaced mesh so
 * the virtual terrain matches what the camera saw.
 */
function buildHeightmapFromFrame(
    cv: any,
    video: HTMLVideoElement,
    centerPx: { x: number; y: number },
    radiusPx: number,
    outSize: number = 96,
): TerrainHeightmap | null {
    if (!cv) return null;
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw || !vh || radiusPx <= 2) return null;

    // Clamp the crop square to the visible frame.
    const cropSize = Math.max(32, Math.min(Math.round(radiusPx * 2), Math.min(vw, vh)));
    const cx = Math.min(Math.max(centerPx.x, cropSize / 2), vw - cropSize / 2);
    const cy = Math.min(Math.max(centerPx.y, cropSize / 2), vh - cropSize / 2);
    const cropX = Math.round(cx - cropSize / 2);
    const cropY = Math.round(cy - cropSize / 2);

    // Render the video into a canvas to read pixel data.
    const canvas = document.createElement('canvas');
    canvas.width = vw;
    canvas.height = vh;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    ctx.drawImage(video, 0, 0, vw, vh);

    // Build the texture data URL from a circular-cropped canvas (transparent outside the circle).
    const texCanvas = document.createElement('canvas');
    texCanvas.width = cropSize;
    texCanvas.height = cropSize;
    const texCtx = texCanvas.getContext('2d');
    if (!texCtx) return null;
    texCtx.save();
    texCtx.beginPath();
    texCtx.arc(cropSize / 2, cropSize / 2, cropSize / 2 - 1, 0, Math.PI * 2);
    texCtx.closePath();
    texCtx.clip();
    texCtx.drawImage(canvas, cropX, cropY, cropSize, cropSize, 0, 0, cropSize, cropSize);
    texCtx.restore();
    const textureUrl = texCanvas.toDataURL('image/png');

    // OpenCV processing pipeline on the cropped region.
    const fullImg = cv.imread(canvas);
    let heightmap: Float32Array | null = null;
    try {
        const roi = fullImg.roi(new cv.Rect(cropX, cropY, cropSize, cropSize));
        const gray = new cv.Mat();
        cv.cvtColor(roi, gray, cv.COLOR_RGBA2GRAY);

        // Contrast-limited adaptive histogram equalization: pulls out subtle shading on the gray print.
        const clahe = new cv.CLAHE(2.5, new cv.Size(8, 8));
        const enhanced = new cv.Mat();
        clahe.apply(gray, enhanced);

        // Mild Gaussian blur to tame noise from camera sensor + print texture before resizing.
        const blurred = new cv.Mat();
        cv.GaussianBlur(enhanced, blurred, new cv.Size(5, 5), 0, 0, cv.BORDER_DEFAULT);

        // Resize to the target grid.
        const resized = new cv.Mat();
        cv.resize(blurred, resized, new cv.Size(outSize, outSize), 0, 0, cv.INTER_AREA);

        // Normalize to [0, 1] using per-sample mean/std — keeps the central "disk plane" at 0.5
        // even when the print is uniformly bright or dark.
        const src = resized.data as Uint8Array; // Mat is 8U single channel at this point.
        heightmap = new Float32Array(outSize * outSize);
        // Only take pixels inside the circle mask when computing stats to avoid the black corners
        // of the crop biasing the normalization.
        let sum = 0;
        let sum2 = 0;
        let count = 0;
        const r2 = (outSize / 2 - 0.5) ** 2;
        for (let y = 0; y < outSize; y++) {
            for (let x = 0; x < outSize; x++) {
                const dx = x - outSize / 2;
                const dy = y - outSize / 2;
                if (dx * dx + dy * dy > r2) continue;
                const v = src[y * outSize + x] / 255;
                sum += v;
                sum2 += v * v;
                count++;
            }
        }
        const mean = count > 0 ? sum / count : 0.5;
        const variance = count > 0 ? Math.max(1e-6, sum2 / count - mean * mean) : 1;
        const std = Math.sqrt(variance);
        for (let y = 0; y < outSize; y++) {
            for (let x = 0; x < outSize; x++) {
                const v = src[y * outSize + x] / 255;
                // Z-score then squash to [0,1] — ±2 std covers almost all pixels.
                const z = (v - mean) / std;
                heightmap[y * outSize + x] = Math.max(0, Math.min(1, 0.5 + z * 0.25));
            }
        }

        roi.delete();
        gray.delete();
        enhanced.delete();
        blurred.delete();
        resized.delete();
        clahe.delete();
    } catch (err) {
        console.warn('[AR][heightmap] pipeline error:', err);
    } finally {
        fullImg.delete();
    }

    if (!heightmap) return null;

    return {
        data: heightmap,
        size: outSize,
        // Detail amplitude in scaled-local units. Pumped up 2.4× from the original 0.25 so
        // dents + ridges feel carved into the terrain, not just softly shaded.
        scaleY: 0.75,
        // Taller cap + √(1−ρ²) profile in sampleHeightmap → stronger “wrapper” over the print.
        domeHeight: 1.65,
        // Gamma > 1 exaggerates strong highlights / shadows so bumps feel bumpier.
        detailGamma: 1.65,
        textureUrl,
        sourceFrameW: vw,
        sourceFrameH: vh,
        centerPx: { x: cx, y: cy },
        radiusPx: cropSize / 2,
        cropRect: { x: cropX, y: cropY, size: cropSize },
    };
}

/**
 * Output of the OpenCV.js gray-surface detector. All pixel values are in the source frame
 * (the AR.js video element's native resolution). `areaFraction` is the largest gray contour's
 * area divided by total frame pixels — a cheap confidence score.
 */
export type GrayDetection = {
    centerX: number;
    centerY: number;
    radiusPx: number;
    frameW: number;
    frameH: number;
    areaFraction: number;
    timestamp: number;
};

/**
 * Run one pass of the gray-silhouette detector on an ImageData buffer.
 * Strategy: downscale → HSV → mask (low saturation + mid value) → morphology → largest contour
 * → minimum enclosing circle. Tunables are kept broad so lighting changes don't kill detection.
 * Returns null if no gray region meets the minimum area threshold.
 */
function detectGrayBlob(cv: any, imageData: ImageData, minAreaFraction = 0.01): GrayDetection | null {
    const src = cv.matFromImageData(imageData);
    const hsv = new cv.Mat();
    const mask = new cv.Mat();
    const lower = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [0, 0, 40, 0]);
    const upper = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [180, 60, 210, 255]);
    const contours = new cv.MatVector();
    const hierarchy = new cv.Mat();
    try {
        cv.cvtColor(src, hsv, cv.COLOR_RGBA2RGB);
        cv.cvtColor(hsv, hsv, cv.COLOR_RGB2HSV);
        // Gray = low saturation, medium-high value; excludes very dark (shadows) and pure white (highlights/specular).
        const low = new cv.Mat(1, 1, cv.CV_8UC3);
        const high = new cv.Mat(1, 1, cv.CV_8UC3);
        low.data.set([0, 0, 40]);
        high.data.set([180, 60, 210]);
        cv.inRange(hsv, low, high, mask);
        low.delete(); high.delete();

        // Close small holes + drop speckle noise.
        const kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(7, 7));
        cv.morphologyEx(mask, mask, cv.MORPH_CLOSE, kernel);
        cv.morphologyEx(mask, mask, cv.MORPH_OPEN, kernel);
        kernel.delete();

        cv.findContours(mask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

        let bestIdx = -1;
        let bestArea = 0;
        for (let i = 0; i < contours.size(); i++) {
            const c = contours.get(i);
            const a = cv.contourArea(c);
            if (a > bestArea) { bestArea = a; bestIdx = i; }
            c.delete();
        }

        const totalPx = mask.rows * mask.cols;
        const areaFraction = bestArea / totalPx;
        if (bestIdx < 0 || areaFraction < minAreaFraction) return null;

        const best = contours.get(bestIdx);
        const { center, radius } = cv.minEnclosingCircle(best);
        best.delete();
        return {
            centerX: center.x,
            centerY: center.y,
            radiusPx: radius,
            frameW: mask.cols,
            frameH: mask.rows,
            areaFraction,
            timestamp: Date.now(),
        };
    } finally {
        src.delete();
        hsv.delete();
        mask.delete();
        lower.delete();
        upper.delete();
        contours.delete();
        hierarchy.delete();
    }
}

function translationFromPose(elements: number[]): MarkerOffset {
    return { x: elements[12], y: elements[13], z: elements[14] };
}

/** Column-major pose inverse-rotate to express the global center in a marker's local frame. */
function centerOffsetInMarkerLocalFromPose(elements: number[], centerGlobal: MarkerOffset): MarkerOffset {
    const tx = elements[12];
    const ty = elements[13];
    const tz = elements[14];
    const dx = centerGlobal.x - tx;
    const dy = centerGlobal.y - ty;
    const dz = centerGlobal.z - tz;
    return {
        x: elements[0] * dx + elements[1] * dy + elements[2] * dz,
        y: elements[4] * dx + elements[5] * dy + elements[6] * dz,
        z: elements[8] * dx + elements[9] * dy + elements[10] * dz,
    };
}

function parsePosesFromConfig(json: any): Record<number, number[]> {
    const controls = Array.isArray(json?.subMarkersControls) ? json.subMarkersControls : [];
    const byId: Record<number, number[]> = {};
    for (const row of controls) {
        const id = row?.parameters?.barcodeValue;
        const pose = row?.poseMatrix;
        if (typeof id !== 'number' || !Array.isArray(pose) || pose.length < 16) continue;
        byId[id] = pose;
    }
    return byId;
}

/** Reads public/surface_pair_report.json (or table_rotation_report.json) marker poses. */
function parsePosesFromReport(json: any): MarkerPoseSource | null {
    const markers = json?.markers;
    if (!markers || typeof markers !== 'object') return null;
    const byId: Record<number, number[]> = {};
    for (const id of ALL_MARKER_IDS) {
        const row = markers[String(id)];
        const pose = row?.poseMatrix;
        if (!Array.isArray(pose) || pose.length < 16) continue;
        byId[id] = pose;
    }
    for (const id of TABLE_MARKER_IDS) {
        if (!byId[id]) return null;
    }
    const c = json?.center_1346_m;
    const center =
        c && typeof c.x === 'number' && typeof c.y === 'number' && typeof c.z === 'number'
            ? { x: c.x, y: c.y, z: c.z }
            : undefined;
    return { byId, center };
}

function setsEqual(a: Set<number>, b: Set<number>): boolean {
    if (a.size !== b.size) return false;
    for (const v of a) {
        if (!b.has(v)) return false;
    }
    return true;
}

/**
 * Calibration-panel row that surfaces the OpenCV.js gray-detection result and, on demand,
 * auto-fits the flat disk radius to it. Mapping pixel → marker units uses a field-of-view
 * heuristic: at a typical phone FoV (~60° horizontal) and the common viewing distance of a
 * marker (~0.4 m), 1 marker unit (0.0508 m ≈ 2") occupies roughly 1/8 of the frame width.
 * The user can always fine-tune with the slider; the button just gets them in the ballpark.
 */
const GrayDiskAutoFitRow: React.FC<{
    cvReady: boolean;
    lastDetection: GrayDetection | null;
    onApply: (radiusInMarkerUnits: number) => void;
}> = ({ cvReady, lastDetection, onApply }) => {
    const fresh = lastDetection && Date.now() - lastDetection.timestamp < 2000;
    const widthFraction = fresh && lastDetection
        ? (2 * lastDetection.radiusPx) / lastDetection.frameW
        : 0;

    return (
        <div style={{
            display: 'flex',
            flexDirection: 'column',
            gap: 4,
            marginTop: 4,
            padding: 6,
            borderRadius: 6,
            background: 'rgba(0, 0, 0, 0.28)',
            border: '1px dashed rgba(123,255,178,0.35)',
        }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10 }}>
                <span style={{ opacity: 0.85 }}>OpenCV gray detection</span>
                <span style={{ color: cvReady ? '#7bffb2' : '#f6a25e', fontWeight: 700 }}>
                    {cvReady ? (fresh ? 'LIVE' : 'idle') : 'loading…'}
                </span>
            </div>
            <div style={{ fontSize: 10, opacity: 0.75 }}>
                {fresh && lastDetection ? (
                    <>
                        Ø {(2 * lastDetection.radiusPx).toFixed(0)}px ({(widthFraction * 100).toFixed(0)}% of frame), area {(lastDetection.areaFraction * 100).toFixed(1)}%
                    </>
                ) : (
                    'Point camera at gray asteroid face to detect…'
                )}
            </div>
            <button
                type="button"
                disabled={!fresh}
                onClick={() => {
                    if (!fresh || !lastDetection) return;
                    // Heuristic mapping: the detected width fills (widthFraction) of the frame.
                    // A 2" marker at typical viewing distance fills ~1/8 of the frame → so
                    // detected_diameter_in_marker_units ≈ widthFraction * 8, radius = half of that.
                    const diameterMarkerUnits = widthFraction * 8;
                    const radiusMarkerUnits = Math.max(1, Math.min(20, diameterMarkerUnits / 2));
                    onApply(radiusMarkerUnits);
                }}
                style={{
                    width: '100%',
                    padding: '6px 8px',
                    borderRadius: 6,
                    border: '1px solid rgba(123,255,178,0.35)',
                    background: fresh ? 'rgba(24, 52, 40, 0.9)' : 'rgba(40, 40, 40, 0.5)',
                    color: fresh ? '#7bffb2' : '#888',
                    fontSize: 10,
                    fontWeight: 700,
                    cursor: fresh ? 'pointer' : 'not-allowed',
                }}
            >
                Auto-fit disk to detected gray
            </button>
        </div>
    );
};

/*
 * Register a custom A-Frame component that renders a displaced CircleGeometry driven by a
 * heightmap + textured with the snapshot. Run once per page load. It reads state via setter
 * methods rather than A-Frame's schema because Float32Array doesn't serialize to an HTML
 * attribute — we stash a token into the attribute and pull the payload from a module-local map.
 */
const _terrainPayloadRegistry = new Map<string, { hm: TerrainHeightmap; radius: number }>();
let _terrainRegistered = false;
function registerHeightmapTerrainComponent() {
    if (_terrainRegistered) return;
    const AFRAME = (window as any).AFRAME;
    const THREE = (window as any).THREE;
    if (!AFRAME || !THREE) return;
    if (AFRAME.components && AFRAME.components['heightmap-terrain']) {
        _terrainRegistered = true;
        return;
    }
    AFRAME.registerComponent('heightmap-terrain', {
        schema: {
            token: { type: 'string', default: '' },
            segments: { type: 'number', default: 64 },
            opacity: { type: 'number', default: 0.95 },
        },
        update: function () {
            const data = this.data;
            const payload = _terrainPayloadRegistry.get(data.token);
            if (!payload) return;
            const { hm, radius } = payload;
            const segs = Math.max(16, data.segments | 0);

            // Square plane — we carve out the circular play zone by collapsing vertices beyond
            // the radius down to the disk plane and pushing them under the ground via alpha.
            const geom = new THREE.PlaneGeometry(radius * 2, radius * 2, segs, segs);
            geom.rotateX(-Math.PI / 2);

            const pos = geom.attributes.position;
            for (let i = 0; i < pos.count; i++) {
                const px = pos.getX(i);
                const pz = pos.getZ(i);
                const r2 = px * px + pz * pz;
                let y = 0;
                if (r2 <= radius * radius) {
                    y = sampleHeightmap(hm, px, pz, radius);
                }
                pos.setY(i, y);
            }
            geom.computeVertexNormals();

            // Build circular alpha-mask texture so the square plane appears as a disk.
            const maskSize = 256;
            const maskCanvas = document.createElement('canvas');
            maskCanvas.width = maskSize;
            maskCanvas.height = maskSize;
            const mctx = maskCanvas.getContext('2d');
            if (mctx) {
                const grad = mctx.createRadialGradient(
                    maskSize / 2, maskSize / 2, maskSize * 0.44,
                    maskSize / 2, maskSize / 2, maskSize / 2,
                );
                grad.addColorStop(0, 'rgba(255,255,255,1)');
                grad.addColorStop(1, 'rgba(255,255,255,0)');
                mctx.fillStyle = grad;
                mctx.fillRect(0, 0, maskSize, maskSize);
            }
            const alphaTex = new THREE.CanvasTexture(maskCanvas);
            alphaTex.needsUpdate = true;

            const colorTex = new THREE.TextureLoader().load(hm.textureUrl);
            colorTex.colorSpace = (THREE as any).SRGBColorSpace ?? (THREE as any).sRGBEncoding;
            colorTex.needsUpdate = true;

            const material = new THREE.MeshStandardMaterial({
                map: colorTex,
                alphaMap: alphaTex,
                transparent: true,
                opacity: data.opacity,
                roughness: 0.85,
                metalness: 0.0,
                side: THREE.DoubleSide,
            });

            // Clean up any prior mesh.
            const prev = this.el.getObject3D('mesh');
            if (prev && prev.geometry) prev.geometry.dispose();
            if (prev && prev.material) {
                if (Array.isArray(prev.material)) prev.material.forEach((m: any) => m.dispose());
                else prev.material.dispose();
            }

            const mesh = new THREE.Mesh(geom, material);
            mesh.frustumCulled = false;
            this.el.setObject3D('mesh', mesh);
        },
        remove: function () {
            const prev = this.el.getObject3D('mesh');
            if (prev && prev.geometry) prev.geometry.dispose();
            if (prev && prev.material) {
                if (Array.isArray(prev.material)) prev.material.forEach((m: any) => m.dispose());
                else prev.material.dispose();
            }
            this.el.removeObject3D('mesh');
        },
    });
    _terrainRegistered = true;
}

const App = () => {
    const [gameState, setGameState] = useState('MENU');

    // AR-mode CSS/JS overrides are scoped to the `ar-active` class on <html>.
    // Without it WEB_GAME uses A-Frame's natural defaults (which work correctly).
    useEffect(() => {
        if (gameState === 'AR_MODE') {
            document.documentElement.classList.add('ar-active');
            return () => { document.documentElement.classList.remove('ar-active'); };
        }
    }, [gameState]);

    // A-Frame 1.6.0 in embedded mode initialises camera.aspect before the
    // canvas CSS dimensions resolve on mobile, leaving it stuck at 1.0 (square)
    // which stretches the scene vertically on portrait. Dispatching window
    // resize alone doesn't always retake — directly set camera.aspect +
    // renderer.setSize on the THREE objects A-Frame exposes, retrying until
    // the scene is fully constructed.
    useEffect(() => {
        if (gameState !== 'WEB_GAME') return;
        const fixAspect = () => {
            const sceneEl = document.querySelector('a-scene') as any;
            if (!sceneEl) return;
            const w = window.innerWidth;
            const h = window.innerHeight;
            if (sceneEl.camera) {
                sceneEl.camera.aspect = w / h;
                sceneEl.camera.updateProjectionMatrix();
            }
            if (sceneEl.renderer) {
                sceneEl.renderer.setSize(w, h, false);
            }
        };
        const timers = [50, 150, 400, 1000, 2500].map(ms =>
            window.setTimeout(fixAspect, ms)
        );
        return () => timers.forEach(window.clearTimeout);
    }, [gameState]);

    const [score, setScore] = useState(0);
    const [difficulty, setDifficulty] = useState<'easy' | 'normal' | 'hard'>('normal');
    const modeCfg = MODE_CONFIG[difficulty as Difficulty];
    // Samples (collectibles) and Obstacles
    const [samples, setSamples] = useState<{ id: string; x: number; y: number; z: number; model: SampleModel; rotation: string }[]>([]);
    const samplesRef = useRef<typeof samples>([]);
    samplesRef.current = samples;
    const [samplesCollected, setSamplesCollected] = useState(0);

    const [obstacles, setObstacles] = useState<{ id: string; x: number; y: number; z: number; radius: number}[]>([]);
    const obstaclesRef = useRef<typeof obstacles>([]);
    obstaclesRef.current = obstacles;

    // Energy meter (0..100) - skeleton only
    const [energy, setEnergy] = useState(MAX_ENERGY);
    const [showDifficulty, setShowDifficulty] = useState(false);
    const [showCredits, setShowCredits] = useState(false);
    const [showIntroPopup, setShowIntroPopup] = useState(false);
    const [introPopupCanClose, setIntroPopupCanClose] = useState(false);
    const showIntroPopupRef = useRef(false);
    showIntroPopupRef.current = showIntroPopup;
    const introLockoutTimerRef = useRef<number | null>(null);
    const [showEndScreen, setShowEndScreen] = useState(false);
    const [endReason, setEndReason] = useState<'complete' | 'energy'>('complete');
    const [energyBonus, setEnergyBonus] = useState(0);

    const [scanPrompt, setScanPrompt] = useState(true);
    const [meshLoaded, setMeshLoaded] = useState(false);
    const [roverReady, setRoverReady] = useState(false);
    const lastDirectionRef = useRef<[number, number]>([0, 1]);
    const keysHeld = useRef(new Set<string>());
    const dpadInputRef = useRef<[number, number]>([0, 0]);
    const moveLoopId = useRef<number | null>(null);
    const lastMoveTime = useRef(0);
    const prevCamUp = useRef<any>(null);
    const energyRef = useRef(MAX_ENERGY);
    energyRef.current = energy;
    const wasInObstacleRef = useRef(false);
    const endTriggeredRef = useRef(false);
    const modeCfgRef = useRef(modeCfg);
    modeCfgRef.current = modeCfg;
    // Keyboard navigation
    const playBtnRef = useRef<HTMLButtonElement | null>(null);
    const arBtnRef = useRef<HTMLButtonElement | null>(null);
    const creditsBtnRef = useRef<HTMLButtonElement | null>(null);
    const diffBtnRefs = [useRef<HTMLButtonElement | null>(null), useRef<HTMLButtonElement | null>(null), useRef<HTMLButtonElement | null>(null)];
    const sampleContinueBtnRef = useRef<HTMLButtonElement | null>(null);
    const [waypointPopup, setWaypointPopup] = useState<{ title: string; body?: string; image?: string; } | null>(null);

    /** ---------- AR calibration + scale constants (pulled from calibrated AR build) ---------- */
    const [centerOffsetsById, setCenterOffsetsById] = useState<Record<number, MarkerOffset>>({});
    const [arVisibleIds, setArVisibleIds] = useState<Set<number>>(new Set());
    const [arAnchorId, setArAnchorId] = useState<number | null>(null);
    const arAnchorHoldTimeoutRef = useRef<number | null>(null);
    const arLastSeenMsRef = useRef<Record<number, number>>({});

    // Parent entity transform that places the asteroid near the table marker.
    const [arCalibration, setArCalibration] = useState<ArCalibration>(() => loadArCalibration());
    // OpenCV.js gray-surface detection — validates that the virtual disk actually covers the
    // gray silhouette of the 3D-printed asteroid. See useEffect below for the detection loop.
    const [cvReady, setCvReady] = useState(false);
    const [grayDetection, setGrayDetection] = useState<GrayDetection | null>(null);

    /*
     * Tap-to-map setup flow. Each AR mission starts in AWAIT_CENTER; one tap on the gray face
     * records the disk center (in scaled-local units), a second tap on the edge records the
     * radius, and we transition to READY → the game spawns + begins. This skips the slider
     * calibration entirely for first-time users.
     */
    type ArSetupPhase = 'AWAIT_CENTER' | 'AWAIT_EDGE' | 'READY';
    const [arSetupPhase, setArSetupPhase] = useState<ArSetupPhase>('AWAIT_CENTER');
    const arSetupPhaseRef = useRef(arSetupPhase);
    useEffect(() => { arSetupPhaseRef.current = arSetupPhase; }, [arSetupPhase]);

    /*
     * Terrain snapshot: at the moment the player finishes tap-to-map, we grab the current video
     * frame, crop the circular play zone, and run it through the OpenCV pipeline to get a
     * heightmap. The heightmap drives rover Y + tilt + sample/obstacle placement. The cropped
     * snapshot becomes the texture on a displaced mesh so the user can see their captured terrain.
     *
     * `terrainToken` is how we bind the React-side payload to the A-Frame custom component
     * (see registerHeightmapTerrainComponent). When the token changes, the component rebuilds.
     */
    const [terrainToken, setTerrainToken] = useState<string | null>(null);
    /** Jet heatmap preview of motion-fused depth (AR flat mode). */
    const [arDepthHeatmapUrl, setArDepthHeatmapUrl] = useState<string | null>(null);
    const terrainRef = useRef<TerrainHeightmap | null>(null);
    // Screen-space pixel coordinates of each tap — needed to crop the correct circle out of the video frame.
    const tapScreenCenterRef = useRef<{ x: number; y: number } | null>(null);
    // Persist any slider change so the next session starts with the tuned values.
    useEffect(() => {
        try {
            localStorage.setItem(AR_CALIBRATION_STORAGE_KEY, JSON.stringify(arCalibration));
        } catch {
            /* storage may be disabled — not fatal */
        }
    }, [arCalibration]);
    const updateArCalibration = useCallback((patch: Partial<ArCalibration>) => {
        setArCalibration((prev) => ({ ...prev, ...patch }));
    }, []);
    const {
        modelLift,
        modelBack,
        modelYawOffsetDeg,
        modelPitchOffsetDeg,
        modelRollOffsetDeg,
        modelScaleX,
        modelScaleY,
        modelScaleZ,
        sampleScaleFr,
        compensateScaleWithLift,
        liftDistancePivot,
        hideVirtualAsteroid,
        showCalibrationCube,
        flatSurfaceMode,
        flatSurfaceRadius,
        flatSurfaceHeight,
        flatSurfaceOffsetX,
        flatSurfaceOffsetZ,
        showFlatDisk,
        hideTerrainSurface,
    } = arCalibration;

    /**
     * Flat AR only: after tap-to-map reaches READY, reveal in order — HUD (score/energy/joystick),
     * then field (disk/terrain/samples/obstacles), then rover. During MAP YOUR ZONE, HUD stays off.
     */
    type ArFlatRevealPhase = 'SETUP' | 'HUD' | 'FIELD' | 'ROVER';
    const [arFlatRevealPhase, setArFlatRevealPhase] = useState<ArFlatRevealPhase>('SETUP');
    const arFlatRevealPhaseRef = useRef(arFlatRevealPhase);
    useEffect(() => { arFlatRevealPhaseRef.current = arFlatRevealPhase; }, [arFlatRevealPhase]);

    useLayoutEffect(() => {
        if (gameState !== 'AR_MODE' || !flatSurfaceMode) {
            setArFlatRevealPhase('ROVER');
            return;
        }
        if (arSetupPhase !== 'READY') {
            setArFlatRevealPhase('SETUP');
            return;
        }
        setArFlatRevealPhase('HUD');
    }, [gameState, flatSurfaceMode, arSetupPhase]);

    useEffect(() => {
        if (gameState !== 'AR_MODE' || !flatSurfaceMode || arSetupPhase !== 'READY') return;
        const t1 = window.setTimeout(() => setArFlatRevealPhase('FIELD'), 550);
        const t2 = window.setTimeout(() => setArFlatRevealPhase('ROVER'), 1200);
        return () => {
            window.clearTimeout(t1);
            window.clearTimeout(t2);
        };
    }, [gameState, flatSurfaceMode, arSetupPhase]);

    /**
     * "Digital-twin" AR pattern: the asteroid collision GLB is kept as the invisible physics surface
     * (rover/crystals/obstacles raycast against it) but the visual is hidden so the user sees the
     * real 3D-printed rock through the camera. Browser-based real-surface detection (WebXR Depth
     * Sensing / 8th Wall mesh / Niantic Lightship) would let us drop the twin entirely, but it
     * requires leaving the AR.js marker stack — flip `hideVirtualAsteroid` off any time to verify
     * alignment between twin and physical rock during calibration.
     */
    const showArAsteroid = !hideVirtualAsteroid;
    /**
     * Toggles a bright green debug sphere co-located with the AR rover. Great for verifying
     * whether the rover's computed position is actually on the asteroid surface when the
     * rover model itself is hard to see/orient.
     */
    const showArRoverDebugSphere = false;

    /** Persisted across anchor switches so the rover stays on the asteroid even when the AR parent remounts. */
    const roverPosRef = useRef<{ x: number; y: number; z: number } | null>(null);

    // Refs for flat-surface tunables so the movement loop always sees the latest tap values.
    const flatSurfaceModeRef = useRef(flatSurfaceMode);
    const flatSurfaceRadiusRef = useRef(flatSurfaceRadius);
    const flatSurfaceHeightRef = useRef(flatSurfaceHeight);
    const flatSurfaceOffsetXRef = useRef(flatSurfaceOffsetX);
    const flatSurfaceOffsetZRef = useRef(flatSurfaceOffsetZ);
    useEffect(() => { flatSurfaceModeRef.current = flatSurfaceMode; }, [flatSurfaceMode]);
    useEffect(() => { flatSurfaceRadiusRef.current = flatSurfaceRadius; }, [flatSurfaceRadius]);
    useEffect(() => { flatSurfaceHeightRef.current = flatSurfaceHeight; }, [flatSurfaceHeight]);
    useEffect(() => { flatSurfaceOffsetXRef.current = flatSurfaceOffsetX; }, [flatSurfaceOffsetX]);
    useEffect(() => { flatSurfaceOffsetZRef.current = flatSurfaceOffsetZ; }, [flatSurfaceOffsetZ]);

    /*
     * OpenCV.js readiness — the script is loaded async from /vendor/opencv.js in index.html.
     * `cv.onRuntimeInitialized` fires once the WASM module is usable; until then, every cv.*
     * call would throw. We poll for it so we don't race the script tag.
     */
    useEffect(() => {
        let cancelled = false;
        const check = () => {
            if (cancelled) return;
            const cv = (window as any).cv;
            if (cv && typeof cv.Mat === 'function') {
                setCvReady(true);
                return;
            }
            if (cv && typeof cv.onRuntimeInitialized !== 'undefined') {
                cv.onRuntimeInitialized = () => {
                    if (!cancelled) setCvReady(true);
                };
                return;
            }
            window.setTimeout(check, 200);
        };
        check();
        return () => { cancelled = true; };
    }, []);

    /*
     * Register the A-Frame heightmap-terrain component once both AFRAME and THREE are on window.
     * A-Frame bundles THREE but loads asynchronously, so we poll briefly.
     */
    useEffect(() => {
        let cancelled = false;
        const tryRegister = () => {
            if (cancelled) return;
            if ((window as any).AFRAME && (window as any).THREE) {
                registerHeightmapTerrainComponent();
                return;
            }
            window.setTimeout(tryRegister, 150);
        };
        tryRegister();
        return () => { cancelled = true; };
    }, []);

    /*
     * Gray-surface detection loop. Runs at ~4 fps while in AR mode to keep the CPU cost low
     * (OpenCV.js is WASM but still not free). Copies the AR video element into a downscaled
     * canvas (320×240), runs detectGrayBlob, and stores the result. The calibration panel
     * reads grayDetection to show diameter + confidence to the user.
     */
    useEffect(() => {
        if (!cvReady) return;
        if (gameState !== 'AR_MODE') return;
        const cv = (window as any).cv;
        if (!cv) return;

        let cancelled = false;
        const canvas = document.createElement('canvas');
        const SCAN_W = 320;
        const SCAN_H = 240;
        canvas.width = SCAN_W;
        canvas.height = SCAN_H;
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        if (!ctx) return;

        const tick = () => {
            if (cancelled) return;
            const video = document.querySelector('video');
            if (video && video.readyState >= 2 && video.videoWidth > 0) {
                try {
                    ctx.drawImage(video, 0, 0, SCAN_W, SCAN_H);
                    const img = ctx.getImageData(0, 0, SCAN_W, SCAN_H);
                    const result = detectGrayBlob(cv, img);
                    if (!cancelled) setGrayDetection(result);
                } catch (err) {
                    // Transient decode errors can happen during marker pose jitter; swallow them.
                    console.warn('[AR][cv] gray detection error:', err);
                }
            }
            if (!cancelled) window.setTimeout(tick, 250);
        };
        tick();
        return () => { cancelled = true; };
    }, [cvReady, gameState]);

    /*
     * Tap-to-map play-area selector.
     *
     * Listens for pointer events on the AR canvas during the setup phases. Each tap is converted
     * from screen NDC → world ray (via the AR.js camera) → intersection with the active marker's
     * local XZ plane → marker-local coordinates → scaled-local coordinates (divided by modelScale
     * so everything else in this file that positions things in the scale transform still works).
     *
     * Phase 1 (AWAIT_CENTER): tap sets (flatSurfaceOffsetX, flatSurfaceOffsetZ).
     * Phase 2 (AWAIT_EDGE):   tap sets flatSurfaceRadius (distance from center to this tap).
     * Phase 3 (READY):        tap handler disarms; sample + obstacle spawn + rover init fire
     *                          because their gating effects see phase === READY.
     *
     */
    useEffect(() => {
        if (gameState !== 'AR_MODE') return;
        if (!flatSurfaceMode) return;
        if (arSetupPhase === 'READY') return;
        if (arAnchorId === null) return;

        const sceneEl = document.querySelector('a-scene') as any;
        const canvas: HTMLCanvasElement | null = sceneEl?.canvas ?? null;
        if (!canvas) return;

        const raycastToPlayPlane = (clientX: number, clientY: number): { x: number; z: number } | null => {
            const THREE = (window as any).THREE;
            if (!THREE || !sceneEl?.camera) return null;

            // Play-root carries the scale transform; its local XZ plane is the flat disk surface.
            // worldToLocal inverts the full chain (marker → centerOffset → scale) for us, so taps
            // come out directly in the same coordinate space the JSX + physics already use.
            const playRoot = document.getElementById('ar-play-root') as any;
            if (!playRoot?.object3D) return null;

            const rect = canvas.getBoundingClientRect();
            const ndc = new THREE.Vector2(
                ((clientX - rect.left) / rect.width) * 2 - 1,
                -((clientY - rect.top) / rect.height) * 2 + 1,
            );
            const raycaster = new THREE.Raycaster();
            raycaster.setFromCamera(ndc, sceneEl.camera);

            playRoot.object3D.updateWorldMatrix(true, false);
            const worldPos = new THREE.Vector3();
            const worldQuat = new THREE.Quaternion();
            playRoot.object3D.getWorldPosition(worldPos);
            playRoot.object3D.getWorldQuaternion(worldQuat);
            const worldUp = new THREE.Vector3(0, 1, 0).applyQuaternion(worldQuat).normalize();
            const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(worldUp, worldPos);

            const hitWorld = new THREE.Vector3();
            if (!raycaster.ray.intersectPlane(plane, hitWorld)) return null;

            const hitLocal = playRoot.object3D.worldToLocal(hitWorld.clone());
            return { x: hitLocal.x, z: hitLocal.z };
        };

        const onPointerDown = async (e: PointerEvent) => {
            // Ignore taps on HTML UI overlays; they already have pointer-events: auto and stop props on their own buttons.
            if ((e.target as HTMLElement)?.closest('.ar-setup-overlay')) return;
            const hit = raycastToPlayPlane(e.clientX, e.clientY);
            if (!hit) return;

            if (arSetupPhase === 'AWAIT_CENTER') {
                // Save screen-space pixels too — we need them to crop the circular region out of
                // the video frame when generating the terrain heightmap on the next tap.
                tapScreenCenterRef.current = { x: e.clientX, y: e.clientY };
                updateArCalibration({ flatSurfaceOffsetX: hit.x, flatSurfaceOffsetZ: hit.z });
                setArSetupPhase('AWAIT_EDGE');
            } else if (arSetupPhase === 'AWAIT_EDGE') {
                const dx = hit.x - flatSurfaceOffsetX;
                const dz = hit.z - flatSurfaceOffsetZ;
                const radius = Math.max(0.4, Math.hypot(dx, dz));
                updateArCalibration({ flatSurfaceRadius: radius });

                // Terrain capture: use the two tap positions to compute the screen-space circle,
                // then process the current video frame into a heightmap. Done synchronously so
                // the rover + samples spawn with the correct Y values immediately.
                try {
                    const cv = (window as any).cv;
                    const video = document.querySelector('video') as HTMLVideoElement | null;
                    const center = tapScreenCenterRef.current;
                    if (cv && video && video.videoWidth > 0 && center) {
                        const canvasRect = canvas.getBoundingClientRect();
                        const scaleX = video.videoWidth / canvasRect.width;
                        const scaleY = video.videoHeight / canvasRect.height;
                        const cxVid = (center.x - canvasRect.left) * scaleX;
                        const cyVid = (center.y - canvasRect.top) * scaleY;
                        const exVid = (e.clientX - canvasRect.left) * scaleX;
                        const eyVid = (e.clientY - canvasRect.top) * scaleY;
                        const rVid = Math.max(16, Math.hypot(exVid - cxVid, eyVid - cyVid));
                        const hm = buildHeightmapFromFrame(cv, video, { x: cxVid, y: cyVid }, rVid, 96);
                        if (hm) {
                            await augmentHeightmapWithDepthFromMotion(cv, video, hm);
                            terrainRef.current = hm;
                            const token = `terrain-${Date.now()}`;
                            _terrainPayloadRegistry.set(token, { hm, radius });
                            // Trim old tokens so the map doesn't grow unbounded across remaps.
                            if (_terrainPayloadRegistry.size > 6) {
                                const oldest = _terrainPayloadRegistry.keys().next().value;
                                if (oldest) _terrainPayloadRegistry.delete(oldest);
                            }
                            setTerrainToken(token);
                            setArDepthHeatmapUrl(hm.depthDebugUrl ?? null);
                        } else {
                            console.warn('[AR][terrain] heightmap build returned null');
                            setArDepthHeatmapUrl(null);
                        }
                    } else {
                        console.warn('[AR][terrain] skipping build — cv/video not ready');
                        setArDepthHeatmapUrl(null);
                    }
                } catch (err) {
                    console.warn('[AR][terrain] capture failed:', err);
                    setArDepthHeatmapUrl(null);
                }

                setArSetupPhase('READY');
            }
        };

        const onPointerDownBound = (ev: Event) => { void onPointerDown(ev as PointerEvent); };
        canvas.addEventListener('pointerdown', onPointerDownBound);
        return () => {
            canvas.removeEventListener('pointerdown', onPointerDownBound);
        };
    }, [gameState, flatSurfaceMode, arSetupPhase, arAnchorId, modelScaleX, modelScaleY, modelScaleZ, flatSurfaceOffsetX, flatSurfaceOffsetZ, updateArCalibration]);

    // Calibration cube: 1 marker unit = printed marker width = 2" (MARKER_SIZE_METERS) on each edge.
    const MARKER_CUBE_REF_SIZE = 1.0;
    const MARKER_CUBE_WIDTH_RATIO = 1.0;
    const MARKER_CUBE_HEIGHT_RATIO = 1.0;
    const MARKER_CUBE_DEPTH_RATIO = 1.0;
    const markerPlaneOffset = 0.0;
    const markerOverlaySize = MARKER_CUBE_REF_SIZE;
    const markerOverlayWidth = markerOverlaySize * MARKER_CUBE_WIDTH_RATIO;
    const markerOverlayHeight = markerOverlaySize * MARKER_CUBE_HEIGHT_RATIO;
    const markerOverlayDepth = markerOverlaySize * MARKER_CUBE_DEPTH_RATIO;
    const markerOverlayShiftX = 0.0;
    const markerOverlayShiftZ = 0.0;

    // Interior sizes expressed as fractions of the reference cube edge.
    /** ×2 vs calibrated `sampleScaleFr` (prior ×4 halved) so AR crystals stay readable but compact. */
    const AR_SAMPLE_SCALE_FR = sampleScaleFr * 2;
    const AR_ARROW_CONE_HEIGHT_FR = 0.0075;
    const AR_ARROW_CONE_RADIUS_FR = 0.004;
    const AR_ARROW_CYL_RADIUS_FR = 0.001;
    const AR_ARROW_CYL_HEIGHT_FR = 0.009;
    const AR_ARROW_CONE_OFFSET_Y_FR = 0.008;
    const AR_ARROW_CYL_OFFSET_Y_FR = 0.0015;
    const AR_ARROW_ORBIT_RADIUS_FR = 0.1 / MARKER_CUBE_WIDTH_RATIO;
    const AR_ARROW_NORMAL_OFFSET_FR = 0.02;
    const AR_COLLECTION_RADIUS_FR = 0.125;
    const AR_ROVER_SURFACE_OFFSET_FR = 0.03;
    // Baseline 0.625 × 1.5, then × AR_ROVER_SCALE_MULTIPLIER for on-print size.
    const AR_ROVER_DESIRED_SCALE_FR = 0.9375 * AR_ROVER_SCALE_MULTIPLIER;

    const arSampleScale = markerOverlaySize * AR_SAMPLE_SCALE_FR;
    const arSampleScaleStr = `${arSampleScale} ${arSampleScale} ${arSampleScale}`;
    const arArrowConeHeight = markerOverlaySize * AR_ARROW_CONE_HEIGHT_FR * AR_ROVER_SCALE_MULTIPLIER;
    const arArrowConeRadiusBottom = markerOverlaySize * AR_ARROW_CONE_RADIUS_FR * AR_ROVER_SCALE_MULTIPLIER;
    const arArrowCylRadius = markerOverlaySize * AR_ARROW_CYL_RADIUS_FR * AR_ROVER_SCALE_MULTIPLIER;
    const arArrowCylHeight = markerOverlaySize * AR_ARROW_CYL_HEIGHT_FR * AR_ROVER_SCALE_MULTIPLIER;
    const arArrowConeY = markerOverlaySize * AR_ARROW_CONE_OFFSET_Y_FR * AR_ROVER_SCALE_MULTIPLIER;
    const arArrowCylY = markerOverlaySize * AR_ARROW_CYL_OFFSET_Y_FR * AR_ROVER_SCALE_MULTIPLIER;
    const arArrowOrbitRadius = markerOverlayWidth * AR_ARROW_ORBIT_RADIUS_FR * AR_ROVER_SCALE_MULTIPLIER;
    const arArrowNormalOffset = markerOverlaySize * AR_ARROW_NORMAL_OFFSET_FR * AR_ROVER_SCALE_MULTIPLIER;
    const arCollectionRadius = markerOverlaySize * AR_COLLECTION_RADIUS_FR * AR_ROVER_SCALE_MULTIPLIER;
    const arSurfaceOffset = markerOverlaySize * AR_ROVER_SURFACE_OFFSET_FR * AR_ROVER_SCALE_MULTIPLIER;
    const arRoverDesiredScale = markerOverlaySize * AR_ROVER_DESIRED_SCALE_FR;
    // Compensate rover scale so it renders at a consistent world size regardless of the parent's non-uniform scale.
    const arRoverScaleStr = `${arRoverDesiredScale / modelScaleX} ${arRoverDesiredScale / modelScaleY} ${arRoverDesiredScale / modelScaleZ}`;

    /**
     * Visual asteroid GLB must use the same local transform Rust applies when building the collision
     * mesh (rust_engine/src/lib.rs: scale_factor 2.5, offset -3.75 / -2.2 / 3.22). WASM positions
     * for rover, samples, and obstacles are already in that baked space.
     *
     * In AR we deliberately render AsteroidPsyche_Collision.glb (the LOW-POLY mesh that Rust
     * raycasts against) rather than the high-poly visual mesh used on web. That guarantees
     * the visual surface and the physics surface are THE SAME geometry — no vertex-drift between
     * high-poly art and low-poly collider — so the rover, samples, and obstacles sit exactly on
     * what the user sees.
     */
    const arAsteroidGltfScale = '2.5 2.5 2.5';
    const arAsteroidGltfPosition = '-3.75 -2.2 3.22';
    const arAsteroidModelSrc = './models/AsteroidPsyche_Collision.glb';
    const markerLostGraceMs = 700;
    const anchorSwitchDebounceMs = 450;

    /** Initialize WASM and load asteroid collision mesh from GLB. */
    useEffect(() => {
        const initRust = async () => {
            try {
                await init();
                const response = await fetch('./models/AsteroidPsyche_Collision.glb');
                const arrayBuffer = await response.arrayBuffer();
                const bytes = new Uint8Array(arrayBuffer);
                await load_collision_mesh(bytes);
                setMeshLoaded(true);
            } catch (e) {
                console.error("Failed to initialize:", e);
            }
        };

        initRust();
    }, []);

    const closeIntroPopup = () => {
        setShowIntroPopup(false);
        setIntroPopupCanClose(false);
    };

    const returnToMenu = () => {
        setGameState('MENU');
        setShowEndScreen(false);
        setShowIntroPopup(false);
        setIntroPopupCanClose(false);
        setWaypointPopup(null);
        setSamplesCollected(0);
        setScore(0);
        energyRef.current = MAX_ENERGY;
        setEnergy(MAX_ENERGY);
        wasInObstacleRef.current = false;
        endTriggeredRef.current = false;
        setEnergyBonus(0);
        popupIndexRef.current = 0;
        setArAnchorId(null);
        setArVisibleIds(new Set());
        roverPosRef.current = null;
    };

    const handleStart = async (mode: string, chosenDifficulty?: 'easy' | 'normal' | 'hard') => {
        if (chosenDifficulty) setDifficulty(chosenDifficulty);

        if (mode === 'web_game') {
            setGameState('WEB_GAME');
            if (introLockoutTimerRef.current) clearTimeout(introLockoutTimerRef.current);
            setShowIntroPopup(true);
            setIntroPopupCanClose(false);
            introLockoutTimerRef.current = window.setTimeout(() => setIntroPopupCanClose(true), 3500);
        } else if (mode === 'ar') {
            // Fresh tap-to-map flow each mission — user picks a new play zone on the gray face.
            setArSetupPhase('AWAIT_CENTER');
            terrainRef.current = null;
            setTerrainToken(null);
            setArDepthHeatmapUrl(null);
            setGameState('AR_MODE');
            // AR Experience launched from the Launch flow uses the same intro/briefing as web.
            if (chosenDifficulty !== undefined) {
                if (introLockoutTimerRef.current) clearTimeout(introLockoutTimerRef.current);
                setShowIntroPopup(true);
                setIntroPopupCanClose(false);
                introLockoutTimerRef.current = window.setTimeout(() => setIntroPopupCanClose(true), 3500);
            } else {
                setShowIntroPopup(false);
                setIntroPopupCanClose(false);
            }
            try {
                await start_ar_session(mode);
            } catch (e) {
                console.error("Failed to start AR session", e);
                // Continue anyway to show AR scene
            }
        }
    };

    /** Builds right/up/normal frame at position using parallel transport for smooth camera orientation. */
    const getCameraFrame = (px: number, py: number, pz: number) => {
        const THREE = (window as any).THREE;
        const normal = new THREE.Vector3(px, py, pz).normalize();

        let up: any;
        if (prevCamUp.current) {
            up = prevCamUp.current.clone();
            up.addScaledVector(normal, -up.dot(normal));

            if (up.lengthSq() < 0.0001) {
                const ref = new THREE.Vector3(0, 1, 0);
                if (Math.abs(normal.dot(ref)) > 0.9) ref.set(0, 0, -1);
                const tmpRight = new THREE.Vector3().crossVectors(ref, normal).normalize();
                up = new THREE.Vector3().crossVectors(normal, tmpRight);
            }
            up.normalize();
        } else {
            const ref = new THREE.Vector3(0, 1, 0);
            if (Math.abs(normal.dot(ref)) > 0.9) ref.set(0, 0, -1);
            const right = new THREE.Vector3().crossVectors(ref, normal).normalize();
            up = new THREE.Vector3().crossVectors(normal, right);
        }

        const right = new THREE.Vector3().crossVectors(up, normal).normalize();
        up = new THREE.Vector3().crossVectors(normal, right).normalize();

        prevCamUp.current = up.clone();
        return { right, up, normal };
    };

    const popups = [
        {
            title: 'One Sample Collected!',
            body: `
            Psyche is an asteroid between Mars and Jupiter and the name of a NASA space mission to visit that asteroid, led by ASU. Psyche is the first mission to a world likely made largely of metal rather than rock or ice.
            `,
            image: "./images/psycherock.jpg"
        },
        {
            title: 'Two Samples Collected',
            body: `
            Judging from data obtained by Earth-based radar and optical telescopes, scientists hypothesize that the asteroid Psyche could be part of the metal-rich interior of a planetesimal that lost its outer rocky shell.
            `,
            image: "./images/psycherock.jpg"
        },
        {
            title: 'Three Samples Collected',
            body: `
            Previously, the consensus of the science community was that asteroid Psyche was almost entirely metal. New data on density, radar properties, and spectral signatures indicate that the asteroid is possibly a mixed metal and rock world.
            `,
            image: "./images/psycherock.jpg"
        },
        {
            title: 'Four Samples Collected',
            body: `
            Humans can’t bore a path to Earth’s metal core – or the cores of the other rocky planets – so visiting Psyche could provide a one-of-a-kind window into the history of violent collisions and accumulation of matter that created planets like our own.
            `,
            image: "./images/psycherock.jpg"
        },
        {
            title: 'Five Samples Collected',
            body: `
            While rocks on Mars, Venus, and Earth are flush with iron oxides, Psyche’s surface – at least when studied from afar – doesn’t seem to feature much of these chemical compounds.
            `,
            image: "./images/psycherock.jpg"
        },
        {
            title: 'Six Samples Collected',
            body: `
            If the asteroid is leftover core material from a planetary building block, scientists look forward to learning how its history resembles and diverges from that of the rocky planets.
            `,
            image: "./images/psycherock.jpg"
        },
        {
            title: 'Seven Samples Collected',
            body: `
            The surface gravity on Psyche is much less than on Earth, and even less than on the Moon. On Psyche, lifting a car would feel as light as lifting a big dog on Earth!
            `,
            image: "./images/psycherock.jpg"
        },
        {
            title: 'Eight Samples Collected',
            body: `
            The Psyche spacecraft includes three instruments: a magnetometer, multispectral imager, and gamma ray and neutron spectrometer.
            `,
            image: "./images/psycherock.jpg"
        },
        {
            title: 'Nine Samples Collected',
            body: `
            Psyche’s magnetometer will look for evidence of an ancient magnetic field at the asteroid Psyche. A residual magnetic field would be strong evidence the asteroid formed from the core of a planetary body.
            `,
            image: "./images/psycherock.jpg"
        },
        {
            title: 'Ten Samples Collected',
            body: `
            The orbiter’s gamma-ray and neutron spectrometer will help scientists determine the chemical elements that make up the asteroid.
            `,
            image: "./images/psycherock.jpg"
        },
        {
            title: 'Eleven Samples Collected',
            body: `
            The spacecraft’s multispectral imager will provide information about the mineral composition of Psyche as well as its topography. 
            `,
            image: "./images/psycherock.jpg"
        },
        {
            title: 'Twelve Samples Collected',
            body: `
            By analyzing the radio waves the spacecraft communicates with, scientists can measure how the asteroid Psyche affects the spacecraft’s orbit. From that information, scientists can determine the asteroid’s rotation, mass, and gravity field.
            `,
            image: "./images/psycherock.jpg"
        },
        {
            title: 'Thirteen Samples Collected',
            body: `
            The Psyche spacecraft will use a special kind of super-efficient propulsion system for the first time beyond the Moon.
            `,
            image: "./images/psycherock.jpg"
        },
        {
            title: 'Fourteen Samples Collected',
            body: `
            Powered by Hall-effect thrusters, Psyche’s solar electric propulsion system harnesses energy from large solar arrays to create electric and magnetic fields.
            `,
            image: "./images/psycherock.jpg"
        },
        {
            title: 'Fifteen Samples Collected',
            body: `
            The electric and magnetic fields accelerate and expel charged atoms, or ions, of a propellant called xenon. The plasma will emit a sci-fi-like blue glow.
            `,
            image: "./images/psycherock.jpg"
        },
        {
            title: 'Sixteen Samples Collected',
            body: `
            Each of Psyche’s four thrusters, which will operate only one at a time, exert at most the same amount of force that one AA battery would exert on the palm of your hand.
            `,
            image: "./images/psycherock.jpg"
        },
        {
            title: 'Seventeen Samples Collected',
            body: `
            Over time, in the frictionless void of space, the spacecraft will slowly and continuously accelerate.
            `,
            image: "./images/psycherock.jpg"
        },
        {
            title: 'Eighteen Samples Collected',
            body: `
            NASA’s Jet Propulsion Laboratory in Southern California, a leader in robotic exploration of the solar system, manages the mission for the agency’s Science Mission Directorate in Washington.
            `,
            image: "./images/psycherock.jpg"
        },
        {
            title: 'Nineteen Samples Collected',
            body: `
            Psyche launched at 10:19 a.m. EDT Friday, October 13, 2023 aboard a SpaceX Falcon Heavy rocket from Launch Pad 39A at NASA’s Kennedy Space Center in Florida.
            `,
            image: "./images/psycherock.jpg"
        },
        {
            title: 'Twenty Samples Collected',
            body: `
            From launch to arrival at the first science orbit around the asteroid, the spacecraft will travel approximately 1.5 billion miles.
            `,
            image: "./images/psycherock.jpg"
        },
        {
            title: 'Twenty Samples Collected',
            body: `
            From launch to arrival at the first science orbit around the asteroid, the spacecraft will travel approximately 1.5 billion miles!
            `,
            image: "./images/psycherock.jpg"
        }
    ];

    const popupIndexRef = useRef(0);

    /** Advances rover one step: projects input onto tangent plane, raycasts to surface, updates position and camera. */
    const moveRover = useCallback((inputX: number, inputY: number) => {
        if (gameState !== 'WEB_GAME' && gameState !== 'AR_MODE') return;
        if (showEndScreen) return;
        if (showIntroPopup) return;
        if (modeCfgRef.current.energyEnabled && energyRef.current <= 0) return;
        // Flat AR staged reveal: no driving until the rover is shown (HUD/field may already be visible).
        if (gameState === 'AR_MODE' && flatSurfaceModeRef.current && arFlatRevealPhaseRef.current !== 'ROVER') return;

        const THREE = (window as any).THREE;
        const roverId = gameState === 'AR_MODE' ? 'ar-rover' : 'rover';
        const rover = document.getElementById(roverId) as any;
        if (!THREE || !rover) return;

        /*
         * AR: the <a-entity id="ar-rover"> is remounted whenever the active marker anchor changes,
         * which resets its DOM position attribute to the JSX default ("0 0 0"). Reading the DOM
         * then would feed (0,0,0) into move_rover_on_asteroid every tick — which is inside the
         * asteroid volume — and the rover would "move freely" around origin instead of wrapping
         * the surface. Use the persisted roverPosRef as the source of truth instead.
         */
        const domPos = rover.getAttribute('position');
        const currentPos = gameState === 'AR_MODE' && roverPosRef.current
            ? roverPosRef.current
            : domPos;
        lastDirectionRef.current = [inputX, inputY];

        /* Screen-space input → movement. Flat AR: project camera screen axes onto the play disk
         * (see computeArFlatDiskMoveXZ). Web / spherical AR: tangent frame from rover radial. */
        const webStepScale = 0.5;
        let moveDir: any;
        if (gameState === 'AR_MODE' && flatSurfaceModeRef.current) {
            const { mx, mz } = computeArFlatDiskMoveXZ(THREE, inputX, inputY, AR_ROVER_SPEED_SCALE);
            moveDir = new THREE.Vector3(mx, 0, mz);
        } else {
            const { right, up } = getCameraFrame(currentPos.x, currentPos.y, currentPos.z);
            moveDir = gameState === 'AR_MODE'
                ? up.clone().multiplyScalar(inputY).addScaledVector(right, inputX).multiplyScalar(AR_ROVER_SPEED_SCALE)
                : up.clone().multiplyScalar(inputY * webStepScale).addScaledVector(right, inputX * webStepScale);
        }
        let obstacleDrainMultiplier = 1.0;

        if(difficulty == 'normal' || difficulty == 'hard') {
            const cx = currentPos.x, cy = currentPos.y, cz = currentPos.z;
            const obs = obstaclesRef.current;
            const isCollidingWithObstacle = obs.some(o => {
                const dx = o.x - cx;
                const dy = o.y - cy;
                const dz = o.z - cz;
                return dx * dx + dy * dy + dz * dz < o.radius * o.radius;
            });

            const speedMultiplier = isCollidingWithObstacle ? 0.5 : 1.0;
            obstacleDrainMultiplier = isCollidingWithObstacle ? 6 : 1.0;

            if (isCollidingWithObstacle && !wasInObstacleRef.current) {
                setScore(s => Math.max(0, s - modeCfgRef.current.obstaclePenalty));
            }
            wasInObstacleRef.current = isCollidingWithObstacle;

            moveDir = moveDir.clone().multiplyScalar(speedMultiplier);
        }
        

        try {
            let px: number;
            let py: number;
            let pz: number;

            /*
             * Flat-surface mode (AR): the physical print only shows one face to the user, so we walk
             * the rover on a horizontal disk anchored to the active marker instead of wrapping the
             * full 3D mesh. No raycasting, no back-side of the asteroid, no depth ambiguity —
             * purely a 2D step in (x, z). Disable `flatSurfaceMode` to fall back to the full-mesh
             * WASM physics used by the web game.
             */
            if (gameState === 'AR_MODE' && flatSurfaceModeRef.current) {
                // Heightmap-driven walk when a terrain was captured; flat fallback otherwise.
                const hm = terrainRef.current;
                if (hm) {
                    const step = stepOnHeightmap(
                        currentPos,
                        moveDir.x,
                        moveDir.z,
                        flatSurfaceRadiusRef.current,
                        flatSurfaceHeightRef.current,
                        flatSurfaceOffsetXRef.current,
                        flatSurfaceOffsetZRef.current,
                        hm,
                    );
                    px = step.x; py = step.y; pz = step.z;
                } else {
                    const step = stepOnFlatDisk(
                        currentPos,
                        moveDir.x,
                        moveDir.z,
                        flatSurfaceRadiusRef.current,
                        flatSurfaceHeightRef.current,
                        flatSurfaceOffsetXRef.current,
                        flatSurfaceOffsetZRef.current,
                    );
                    px = step.x; py = step.y; pz = step.z;
                }
            } else {
                const result = move_rover_on_asteroid(
                    moveDir.x, moveDir.y, moveDir.z,
                    currentPos.x, currentPos.y, currentPos.z
                );
                if (gameState === 'AR_MODE') {
                    const pushed = pushOutFromCenter(result.position[0], result.position[1], result.position[2], arSurfaceOffset);
                    px = pushed[0]; py = pushed[1]; pz = pushed[2];
                } else {
                    px = result.position[0]; py = result.position[1]; pz = result.position[2];
                }
            }

            rover.setAttribute('position', { x: px, y: py, z: pz });
            // Remember the last surface position so we can restore it if the AR anchor switches markers.
            if (gameState === 'AR_MODE') {
                roverPosRef.current = { x: px, y: py, z: pz };
            }

            if (gameState === 'AR_MODE' && flatSurfaceModeRef.current) {
                // Flat AR: keep the rover level (+Y up in marker space) so the deck / top is always
                // readable from a typical phone angle. Heightmap still drives Y; we do not pitch the
                // body to match local terrain slope.
                updateRoverRotation(rover, 0, 1, 0, moveDir.x, 0, moveDir.z);
            } else {
                updateRoverRotation(rover, px, py, pz, moveDir.x, moveDir.y, moveDir.z);
            }
            // The follow camera only exists in the web scene; AR uses the real-world camera.
            if (gameState !== 'AR_MODE') {
                updateCamera(px, py, pz);
            }

            /* Update sample indicator arrow. */
            const arrowEl = document.getElementById('sample-arrow') as any;
            if (arrowEl) {
                const currentSamples = samplesRef.current;
                const arArrowHold =
                    gameState === 'AR_MODE'
                    && flatSurfaceModeRef.current
                    && arFlatRevealPhaseRef.current !== 'FIELD'
                    && arFlatRevealPhaseRef.current !== 'ROVER';
                if (arArrowHold) {
                    arrowEl.setAttribute('visible', 'false');
                } else if (currentSamples.length === 0) {
                    arrowEl.setAttribute('visible', 'false');
                } else {
                    const rx2 = px, ry2 = py, rz2 = pz;
                    let nearest = currentSamples[0];
                    let nearestDist2 = Infinity;
                    for (const s of currentSamples) {
                        const dx = s.x - rx2, dy = s.y - ry2, dz = s.z - rz2;
                        const d2 = dx * dx + dy * dy + dz * dz;
                        if (d2 < nearestDist2) { nearestDist2 = d2; nearest = s; }
                    }

                    const roverVec = new THREE.Vector3(rx2, ry2, rz2);
                    const normal = roverVec.clone().normalize();

                    // Tangent-plane direction toward nearest sample
                    const toSample = new THREE.Vector3(nearest.x - rx2, nearest.y - ry2, nearest.z - rz2).normalize();
                    const projected = toSample.clone().addScaledVector(normal, -toSample.dot(normal)).normalize();

                    if (projected.lengthSq() > 0.001) {
                        // Orbit: place arrow at fixed radius around rover in the sample's direction.
                        // In AR the orbit/normal offsets scale with the marker cube so the arrow fits the smaller world.
                        const orbitRadius = gameState === 'AR_MODE' ? arArrowOrbitRadius : 0.20;
                        const normalOffset = gameState === 'AR_MODE' ? arArrowNormalOffset : 0.04;
                        const arrowPos = roverVec.clone()
                            .addScaledVector(projected, orbitRadius)
                            .addScaledVector(normal, normalOffset);
                        arrowEl.setAttribute('position', `${arrowPos.x} ${arrowPos.y} ${arrowPos.z}`);

                        // Align arrow apex (+Y) with the projected direction (pointing away from rover)
                        const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), projected);
                        const e = new THREE.Euler().setFromQuaternion(q, 'YXZ');
                        arrowEl.setAttribute('rotation', {
                            x: e.x * 180 / Math.PI,
                            y: e.y * 180 / Math.PI,
                            z: e.z * 180 / Math.PI,
                        });
                    }

                    arrowEl.setAttribute('visible', 'true');
                }
            }

            /* Drain energy on successful movement tick. */
            if (modeCfgRef.current.energyEnabled) {
                const drained = Math.max(0, energyRef.current - modeCfgRef.current.energyDrainPerSec * obstacleDrainMultiplier * (MOVE_INTERVAL / 1000));
                energyRef.current = drained;
                setEnergy(drained);
            }

            /* Check sample collection within radius. AR uses a marker-scaled radius. */
            const COLLECTION_RADIUS = gameState === 'AR_MODE' ? arCollectionRadius : 0.25;
            const rx = px, ry = py, rz = pz;
            const sps = samplesRef.current;
            const collectedSamples = sps.filter(s => {
                const dx = s.x - rx, dy = s.y - ry, dz = s.z - rz;
                return dx * dx + dy * dy + dz * dz < COLLECTION_RADIUS * COLLECTION_RADIUS;
            });
            if (collectedSamples.length > 0) {
                setSamples(prev => prev.filter(s => !collectedSamples.find(c => c.id === s.id)));
                setSamplesCollected(c => c + collectedSamples.length);
                setScore(s => s + collectedSamples.length * modeCfgRef.current.samplePoints);
                const idx = Math.min(popupIndexRef.current, popups.length - 1);
                const popup = popups[idx];
                if (popup) setWaypointPopup(popup);
                popupIndexRef.current += collectedSamples.length;
            }

        } catch (e) {
            console.error("Movement error:", e);
        }
    }, [gameState, difficulty, showEndScreen, showIntroPopup, arSurfaceOffset, arArrowOrbitRadius, arArrowNormalOffset, arCollectionRadius]);

    /**
     * Global keyboard handlers
     */
    useEffect(() => {
        const onKey = (e: KeyboardEvent) => {
            if (waypointPopup) {
                if (e.key === 'Escape' || e.key === 'Enter' || e.key === ' ' || e.code === 'Space') {
                    e.preventDefault();
                    setWaypointPopup(null);
                    return;
                }
            }
            if (showIntroPopup && introPopupCanClose) {
                if (e.key === 'Escape' || e.key === 'Enter' || e.key === ' ' || e.code === 'Space') {
                    e.preventDefault();
                    closeIntroPopup();
                    return;
                }
            }
            if (e.key === 'Escape' || e.key === 'Enter') {
                if (showDifficulty) setShowDifficulty(false);
                if (showCredits) setShowCredits(false);
            }
        };

        window.addEventListener('keydown', onKey);

        return () => window.removeEventListener('keydown', onKey);
    }, [waypointPopup, setWaypointPopup, showDifficulty, showCredits, showIntroPopup, introPopupCanClose]);

    useEffect(() => {
        if (showDifficulty) {
            // focus first difficulty button when opening
            setTimeout(() => diffBtnRefs[0].current?.focus(), 50);
        } else {
            // return focus to Launch Mission button when closing
            setTimeout(() => playBtnRef.current?.focus(), 50);
        }
    }, [showDifficulty]);

    useEffect(() => {
        if (waypointPopup) {
            const t = setTimeout(() => sampleContinueBtnRef.current?.focus(), 50);
            return () => clearTimeout(t);
        }
    }, [waypointPopup]);

    const creditsOpenedOnce = useRef(false);
    useEffect(() => {
        if (showCredits) {
            creditsOpenedOnce.current = true;
        } else if (creditsOpenedOnce.current) {
            // return focus to Credits button when closing (not on initial mount)
            setTimeout(() => creditsBtnRef.current?.focus(), 50);
        }
    }, [showCredits]);

    /**
     * Trap Tab focus inside the start screen when on MENU and modal is closed.
     * This prevents Tab from moving focus out of the app's start UI.
     */
    useEffect(() => {
        if (gameState !== 'MENU' || showDifficulty || showCredits) return;
        const onKey = (e: KeyboardEvent) => {
            if (e.key !== 'Tab') return;
            e.preventDefault();
            const order: HTMLElement[] = [];
            if (playBtnRef.current && !playBtnRef.current.hasAttribute('disabled')) order.push(playBtnRef.current);
            if (arBtnRef.current) order.push(arBtnRef.current);
            if (creditsBtnRef.current) order.push(creditsBtnRef.current);
            if (order.length === 0) return;

            const active = document.activeElement as HTMLElement;
            const idx = order.indexOf(active);
            const dir = e.shiftKey ? -1 : 1;
            let next: number;
            if (idx === -1) {
                next = dir === 1 ? 0 : order.length - 1;
            } else {
                next = (idx + dir + order.length) % order.length;
            }
            order[next].focus();
        };

        window.addEventListener('keydown', onKey);
        return () => window.removeEventListener('keydown', onKey);
    }, [gameState, showDifficulty, showCredits]);

    /** Aligns rover to surface normal with forward direction projected onto tangent plane. */
    const updateRoverRotation = (rover: any, x: number, y: number, z: number, dirX: number, dirY: number, dirZ: number) => {
        const THREE = (window as any).THREE;
        if (!THREE || !rover.object3D) return;

        const surfaceNormal = new THREE.Vector3(x, y, z).normalize();

        /* Project movement direction onto tangent plane. */
        const forward = new THREE.Vector3(dirX, dirY, dirZ);
        forward.addScaledVector(surfaceNormal, -forward.dot(surfaceNormal));
        if (forward.length() < 0.001) return;
        forward.normalize();

        const right = new THREE.Vector3().crossVectors(forward, surfaceNormal).normalize();

        const matrix = new THREE.Matrix4();
        matrix.makeBasis(right, surfaceNormal, forward.clone().multiplyScalar(-1));

        const quaternion = new THREE.Quaternion().setFromRotationMatrix(matrix);
        rover.object3D.quaternion.copy(quaternion);
    };

    const CAMERA_HEIGHT = 2.0;
    const CAMERA_BEHIND = 1.2;

    /** Positions follow camera behind and above rover; look target offset toward asteroid center. */
    const updateCamera = (roverX: number, roverY: number, roverZ: number) => {
        const THREE = (window as any).THREE;
        const cam = document.getElementById('follow-camera') as any;
        if (!THREE || !cam?.object3D) return;

        const { up, normal } = getCameraFrame(roverX, roverY, roverZ);

        const roverPos = new THREE.Vector3(roverX, roverY, roverZ);
        const camPos = roverPos.clone()
            .addScaledVector(normal, CAMERA_HEIGHT)
            .addScaledVector(up, -CAMERA_BEHIND);

        const lookTarget = roverPos.clone().addScaledVector(roverPos.clone().negate(), 0.35);
        const forward = lookTarget.clone().sub(camPos).normalize();
        const camRight = new THREE.Vector3().crossVectors(forward, normal).normalize();
        const camUp = new THREE.Vector3().crossVectors(camRight, forward).normalize();

        cam.object3D.position.set(camPos.x, camPos.y, camPos.z);
        const m = new THREE.Matrix4().makeBasis(camRight, camUp, forward.clone().negate());
        cam.object3D.quaternion.setFromRotationMatrix(m);
    };

    /** Movement loop: merges keyboard and D-pad input, throttles to ~30 moves/sec. */
    const movementLoop = useCallback((timestamp: number) => {
        if (timestamp - lastMoveTime.current >= MOVE_INTERVAL) {
            lastMoveTime.current = timestamp;

            const k = keysHeld.current;
            const [padX, padY] = dpadInputRef.current;
            let inputX = padX;
            let inputY = padY;
            if (k.has('w') || k.has('arrowup')) inputY += 1;
            if (k.has('s') || k.has('arrowdown')) inputY -= 1;
            if (k.has('a') || k.has('arrowleft')) inputX -= 1;
            if (k.has('d') || k.has('arrowright')) inputX += 1;

            inputX = Math.max(-1, Math.min(1, inputX));
            inputY = Math.max(-1, Math.min(1, inputY));

            if (inputX !== 0 || inputY !== 0) moveRover(inputX, inputY);
        }

        const hasKeys = keysHeld.current.size > 0;
        const hasPad = dpadInputRef.current[0] !== 0 || dpadInputRef.current[1] !== 0;
        if (hasKeys || hasPad) {
            moveLoopId.current = requestAnimationFrame(movementLoop);
        } else {
            moveLoopId.current = null;
        }
    }, [moveRover]);

    /** Maps pointer position in circle to normalized input vector; center is dead zone. */
    const updateDpadFromPointer = useCallback((e: React.PointerEvent) => {
        const el = e.currentTarget;
        const rect = el.getBoundingClientRect();
        const cx = rect.left + rect.width / 2;
        const cy = rect.top + rect.height / 2;
        const dx = e.clientX - cx;
        const dy = e.clientY - cy;
        const radius = Math.min(rect.width, rect.height) / 2;
        const deadZone = radius * 0.2;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < deadZone) {
            dpadInputRef.current = [0, 0];
        } else {
            const mag = Math.min(1, (dist - deadZone) / (radius - deadZone));
            const inputX = (dx / dist) * mag;
            const inputY = -(dy / dist) * mag;
            dpadInputRef.current = [inputX, inputY];
        }
        if (moveLoopId.current === null) {
            moveLoopId.current = requestAnimationFrame(movementLoop);
        }
    }, [movementLoop]);

    const clearDpadInput = useCallback(() => {
        dpadInputRef.current = [0, 0];
    }, []);

    /** On game start: reset state and spawn samples/obstacles for both web and AR missions. */
    useEffect(() => {
        const THREE = (window as any).THREE;
        if (gameState === 'WEB_GAME' || gameState === 'AR_MODE') {
            setRoverReady(false);
            setScore(0);
            popupIndexRef.current = 0;
            setWaypointPopup(null);
            prevCamUp.current = null;
            // Force fresh surface spawn when entering a new mission.
            roverPosRef.current = null;
            // AR + flat mode: wait for the user to map the play area via taps before spawning anything.
            const waitingForTapSetup = gameState === 'AR_MODE' && flatSurfaceMode && arSetupPhase !== 'READY';
            if (waitingForTapSetup) {
                setSamples([]);
                setObstacles([]);
                return;
            }
            if (meshLoaded && (gameState === 'WEB_GAME' || gameState === 'AR_MODE')) {
                // Flat-surface AR spawn: place obstacles + samples on a 2D disk in the marker frame,
                // so everything sits on the visible face of the printed asteroid instead of wrapping
                // around an invisible back side. Web game still uses the full 3D mesh physics.
                const flatAr = gameState === 'AR_MODE' && flatSurfaceMode;
                const diskRadius = flatSurfaceRadius;
                /** Cap spawn spread so props stay within ~10\" of the rover on the scanned disk. */
                const clusterRadiusMarker = AR_FLAT_SPAWN_CLUSTER_RADIUS_M / MARKER_SIZE_METERS;
                const playRadius = Math.min(diskRadius * AR_FLAT_PLAY_INNER_RADIUS_FR, clusterRadiusMarker);
                const diskHeight = flatSurfaceHeight;
                const diskOffsetX = flatSurfaceOffsetX;
                const diskOffsetZ = flatSurfaceOffsetZ;

                // Obstacles — spawned first so sample placement can avoid them
                const obsList: { id: string; x: number; y: number; z: number; radius: number }[] = [];
                if (flatAr) {
                    // A few obstacles scattered on the disk; keep them inside the same inner ring as flat movement.
                    // When a terrain heightmap is captured, lift each obstacle to the local surface height
                    // so it sits inside dents / on top of bumps instead of floating at the disk plane.
                    const obsCount = Math.min(OBSTACLE_DIRECTIONS.length, 6);
                    const obsPlacementRadius = playRadius * 0.85;
                    const hmObs = terrainRef.current;
                    for (let i = 0; i < obsCount; i++) {
                        const [, , , baseRadius] = OBSTACLE_DIRECTIONS[i % OBSTACLE_DIRECTIONS.length];
                        const { x, z } = randomPointOnDisk(obsPlacementRadius);
                        const yOff = hmObs ? sampleHeightmap(hmObs, x, z, diskRadius) : 0;
                        obsList.push({
                            id: `o-${i}`,
                            x: diskOffsetX + x,
                            y: diskHeight + yOff - AR_FLAT_SURFACE_Y_SINK,
                            z: diskOffsetZ + z,
                            radius: baseRadius * 0.5,
                        });
                    }
                } else {
                    for (let i = 0; i < OBSTACLE_DIRECTIONS.length; i++) {
                        const [dx, dy, dz, radius] = OBSTACLE_DIRECTIONS[i % OBSTACLE_DIRECTIONS.length];
                        try {
                            const r = get_surface_point_in_direction(dx, dy, dz);
                            obsList.push({ id: `o-${i}`, x: r.position[0], y: r.position[1], z: r.position[2], radius });
                        } catch (_) {}
                    }
                }
                setObstacles(obsList);

                // Samples — randomly placed on the surface, skipping obstacle zones
                const sampleList: { id: string; x: number; y: number; z: number; model: SampleModel; rotation: string }[] = [];
                const MIN_SAMPLE_SPACING = flatAr ? playRadius * 0.22 : 1.5;
                const MAX_ATTEMPTS = modeCfg.spawnSamples * 100;

                // Build a shuffled queue of model types (6 crystal / 7 ore / 7 rock for n=20)
                const n = modeCfg.spawnSamples;
                const base = Math.floor(n / 3);
                const extra = n % 3;
                const modelQueue: SampleModel[] = [
                    ...Array<SampleModel>(base).fill('crystal'),
                    ...Array<SampleModel>(base + (extra >= 1 ? 1 : 0)).fill('ore'),
                    ...Array<SampleModel>(base + (extra >= 2 ? 1 : 0)).fill('rock'),
                ];
                for (let i = modelQueue.length - 1; i > 0; i--) {
                    const j = Math.floor(Math.random() * (i + 1));
                    [modelQueue[i], modelQueue[j]] = [modelQueue[j], modelQueue[i]];
                }

                let attempts = 0;
                while (sampleList.length < modeCfg.spawnSamples && attempts < MAX_ATTEMPTS) {
                    attempts++;
                    try {
                        let px: number, py: number, pz: number;
                        if (flatAr) {
                            const pt = randomPointOnDisk(playRadius);
                            const hmS = terrainRef.current;
                            const yOff = hmS ? sampleHeightmap(hmS, pt.x, pt.z, diskRadius) : 0;
                            px = diskOffsetX + pt.x;
                            py = diskHeight + yOff - AR_FLAT_SURFACE_Y_SINK;
                            pz = diskOffsetZ + pt.z;
                        } else {
                            const dir = randomUnitVector();
                            const r = get_surface_point_in_direction(dir[0], dir[1], dir[2]);
                            px = r.position[0]; py = r.position[1]; pz = r.position[2];
                        }
                        const insideObstacle = obsList.some(o => {
                            const dx = px - o.x;
                            const dy = py - o.y;
                            const dz = pz - o.z;
                            return dx * dx + dy * dy + dz * dz < o.radius * o.radius;
                        });
                        const tooClose = sampleList.some(s => {
                            const dx = px - s.x;
                            const dy = py - s.y;
                            const dz = pz - s.z;
                            return dx * dx + dy * dy + dz * dz < MIN_SAMPLE_SPACING * MIN_SAMPLE_SPACING;
                        });
                        if (!insideObstacle && !tooClose) {
                            let rotation: string;
                            if (flatAr) {
                                // Align "up" with the heightmap normal when we have one, so crystals
                                // tilt the way the ground tilts. Without a heightmap: straight up.
                                const hmR = terrainRef.current;
                                if (hmR) {
                                    const n = sampleHeightmapNormal(hmR, px - diskOffsetX, pz - diskOffsetZ, diskRadius);
                                    const up = new THREE.Vector3(n.nx, n.ny, n.nz);
                                    const alignQ = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), up);
                                    const yawQ = new THREE.Quaternion().setFromAxisAngle(up, Math.random() * Math.PI * 2);
                                    alignQ.premultiply(yawQ);
                                    const e = new THREE.Euler().setFromQuaternion(alignQ, 'YXZ');
                                    const R2D = 180 / Math.PI;
                                    rotation = `${e.x * R2D} ${e.y * R2D} ${e.z * R2D}`;
                                } else {
                                    const yawDeg = Math.random() * 360;
                                    rotation = `0 ${yawDeg} 0`;
                                }
                            } else {
                                // Align local +Y with surface normal (approximated by position-from-origin).
                                const normal = new THREE.Vector3(px, py, pz).normalize();
                                const alignQ = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), normal);
                                const yawQ = new THREE.Quaternion().setFromAxisAngle(normal, Math.random() * Math.PI * 2);
                                alignQ.premultiply(yawQ);
                                const e = new THREE.Euler().setFromQuaternion(alignQ, 'YXZ');
                                const R2D = 180 / Math.PI;
                                rotation = `${e.x * R2D} ${e.y * R2D} ${e.z * R2D}`;
                            }
                            sampleList.push({ id: `s-${sampleList.length}`, x: px, y: py, z: pz, model: modelQueue[sampleList.length], rotation });
                        }
                    } catch (_) { }
                }
                setSamples(sampleList);
                const arrowElStart = document.getElementById('sample-arrow') as any;
                if (arrowElStart) arrowElStart.setAttribute('visible', 'true');

                energyRef.current = MAX_ENERGY;
                setEnergy(MAX_ENERGY);
                wasInObstacleRef.current = false;
                endTriggeredRef.current = false;
            } else {
                setSamples([]);
                setObstacles([]);
                const arrowElStop = document.getElementById('sample-arrow') as any;
                if (arrowElStop) arrowElStop.setAttribute('visible', 'false');
            }
        }
    // Respawn when:
    //   • gameState transitions (web/AR entry)
    //   • the Rust mesh finishes loading
    //   • user toggles flatSurfaceMode
    //   • tap-to-map setup completes (arSetupPhase === READY) → spawn onto the chosen disk
    }, [gameState, meshLoaded, flatSurfaceMode, arSetupPhase, terrainToken]);

    /** Trigger end screen when all samples collected or energy depleted (web or AR). */
    useEffect(() => {
        if ((gameState !== 'WEB_GAME' && gameState !== 'AR_MODE') || endTriggeredRef.current) return;
        if (samplesCollected >= modeCfg.spawnSamples) {
            endTriggeredRef.current = true;
            const bonus = modeCfg.energyBonusEnabled ? Math.round((energyRef.current / MAX_ENERGY) * 1000) : 0;
            setEnergyBonus(bonus);
            setScore(s => s + bonus);
            setEndReason('complete');
            setShowEndScreen(true);
        } else if (modeCfg.energyEnabled && energy <= 0) {
            endTriggeredRef.current = true;
            setEnergyBonus(0);
            setEndReason('energy');
            setShowEndScreen(true);
        }
    }, [samplesCollected, energy, gameState]);

    /** Keyboard listeners and rover init: snap to surface before revealing scene. */
    useEffect(() => {
        if ((gameState !== 'WEB_GAME' && gameState !== 'AR_MODE') || !meshLoaded) {
            return () => { };
        }
        // AR + flat mode: delay rover init until the player maps the play area.
        if (gameState === 'AR_MODE' && flatSurfaceMode && arSetupPhase !== 'READY') {
            return () => { };
        }
        let cancelled = false;
        let retryTimer: number | null = null;

        const scheduleRetry = () => {
            if (cancelled) return;
            retryTimer = window.setTimeout(initRover, 100);
        };

        const initRover = () => {
            if (cancelled) return;
            const roverId = gameState === 'AR_MODE' ? 'ar-rover' : 'rover';
            const rover = document.getElementById(roverId) as any;
            if (!rover) {
                scheduleRetry();
                return;
            }

            try {
                let px: number, py: number, pz: number;
                if (gameState === 'AR_MODE') {
                    if (roverPosRef.current) {
                        // Anchor switched (or re-init): keep the rover where it was.
                        ({ x: px, y: py, z: pz } = roverPosRef.current);
                    } else if (flatSurfaceMode) {
                        // Flat-disk AR spawn: rover lands at the user-tapped disk center. If a
                        // heightmap was captured, lift to the terrain's Y at that spot so the
                        // rover sits flush with the (possibly bumpy) surface instead of floating.
                        px = flatSurfaceOffsetX;
                        const hmInit = terrainRef.current;
                        const yOffset = hmInit ? sampleHeightmap(hmInit, 0, 0, flatSurfaceRadius) : 0;
                        py = flatSurfaceHeight + yOffset - AR_FLAT_SURFACE_Y_SINK;
                        pz = flatSurfaceOffsetZ;
                        roverPosRef.current = { x: px, y: py, z: pz };
                    } else {
                        // Legacy: first-time AR spawn on the full asteroid surface relative to marker anchor.
                        const result = get_surface_point_in_direction(
                            AR_ROVER_START_DIRECTION[0],
                            AR_ROVER_START_DIRECTION[1],
                            AR_ROVER_START_DIRECTION[2]
                        );
                        [px, py, pz] = pushOutFromCenter(result.position[0], result.position[1], result.position[2], arSurfaceOffset);
                        roverPosRef.current = { x: px, y: py, z: pz };
                    }
                } else {
                    const pos = rover.getAttribute('position');
                    const result = move_rover_on_asteroid(0, 0, 0, pos.x, pos.y, pos.z);
                    [px, py, pz] = [result.position[0], result.position[1], result.position[2]];
                }

                rover.setAttribute('position', { x: px, y: py, z: pz });

                /*
                 * Orient the rover to the surface normal BEFORE the user starts moving. If we pass
                 * a zero dir (no input yet) updateRoverRotation bails, leaving the rover in its
                 * JSX default orientation — which looks like it's "floating off" the surface in AR
                 * (no follow-camera to hide it). In flat mode the normal is constant +Y; in mesh
                 * mode we use the camera-frame "up" as a sensible default forward direction.
                 */
                if (gameState === 'AR_MODE' && flatSurfaceMode) {
                    // Level spawn: same +Y-up orientation as flat AR movement (roof visible).
                    updateRoverRotation(rover, 0, 1, 0, 0, 0, 1);
                } else {
                    const { up } = getCameraFrame(px, py, pz);
                    updateRoverRotation(rover, px, py, pz, up.x, up.y, up.z);
                }
                if (gameState !== 'AR_MODE') {
                    updateCamera(px, py, pz);
                }

                setRoverReady(true);
            } catch (e) {
                console.error("Rover init failed:", e);
                scheduleRetry();
            }
        };

        const t = setTimeout(initRover, 50);

        const validKeys = new Set(['w', 'a', 's', 'd', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright']);

        const onKeyDown = (e: KeyboardEvent) => {
            if (showIntroPopupRef.current) return;
            const key = e.key.toLowerCase();
            if (!validKeys.has(key)) return;
            e.preventDefault();
            keysHeld.current.add(key);
            if (moveLoopId.current === null) {
                moveLoopId.current = requestAnimationFrame(movementLoop);
            }
        };

        const onKeyUp = (e: KeyboardEvent) => {
            keysHeld.current.delete(e.key.toLowerCase());
        };

        window.addEventListener('keydown', onKeyDown);
        window.addEventListener('keyup', onKeyUp);

        return () => {
            cancelled = true;
            clearTimeout(t);
            if (retryTimer !== null) window.clearTimeout(retryTimer);
            window.removeEventListener('keydown', onKeyDown);
            window.removeEventListener('keyup', onKeyUp);
            if (moveLoopId.current !== null) {
                cancelAnimationFrame(moveLoopId.current);
                moveLoopId.current = null;
            }
            keysHeld.current.clear();
        };
    }, [gameState, meshLoaded, movementLoop, arSurfaceOffset, flatSurfaceMode, arSetupPhase]);

    /**
     * AR: when the active marker anchor changes, the <a-entity id="ar-rover"> is remounted in a new
     * subtree with its JSX position ("0 0 0"). Re-apply the stored surface position / orientation so
     * the rover stays glued to the asteroid instead of drifting back to the parent origin.
     */
    useEffect(() => {
        if (gameState !== 'AR_MODE') return;
        if (arAnchorId === null) return;
        let cancelled = false;
        let attempts = 0;
        const apply = () => {
            if (cancelled) return;
            const rover = document.getElementById('ar-rover') as any;
            if (!rover || !rover.object3D) {
                attempts++;
                if (attempts < 60) window.setTimeout(apply, 50);
                return;
            }
            try {
                let px: number, py: number, pz: number;
                if (roverPosRef.current) {
                    ({ x: px, y: py, z: pz } = roverPosRef.current);
                } else if (flatSurfaceModeRef.current) {
                    px = flatSurfaceOffsetXRef.current;
                    const hmAR = terrainRef.current;
                    const yOff = hmAR ? sampleHeightmap(hmAR, 0, 0, flatSurfaceRadiusRef.current) : 0;
                    py = flatSurfaceHeightRef.current + yOff - AR_FLAT_SURFACE_Y_SINK;
                    pz = flatSurfaceOffsetZRef.current;
                    roverPosRef.current = { x: px, y: py, z: pz };
                } else {
                    const result = get_surface_point_in_direction(
                        AR_ROVER_START_DIRECTION[0],
                        AR_ROVER_START_DIRECTION[1],
                        AR_ROVER_START_DIRECTION[2]
                    );
                    [px, py, pz] = pushOutFromCenter(result.position[0], result.position[1], result.position[2], arSurfaceOffset);
                    roverPosRef.current = { x: px, y: py, z: pz };
                }
                rover.setAttribute('position', { x: px, y: py, z: pz });
                const [ix, iy] = lastDirectionRef.current;
                if (flatSurfaceModeRef.current) {
                    const hasInput = Math.abs(ix) > 1e-6 || Math.abs(iy) > 1e-6;
                    const fx = hasInput ? ix : 0;
                    const fz = hasInput ? iy : 1;
                    updateRoverRotation(rover, 0, 1, 0, fx, 0, fz);
                } else {
                    const { right, up } = getCameraFrame(px, py, pz);
                    const hasInput = Math.abs(ix) > 1e-6 || Math.abs(iy) > 1e-6;
                    const dir = hasInput
                        ? up.clone().multiplyScalar(iy).addScaledVector(right, ix)
                        : up.clone();
                    updateRoverRotation(rover, px, py, pz, dir.x, dir.y, dir.z);
                }
                setRoverReady(true);
            } catch (e) {
                console.error("AR rover re-snap failed:", e);
            }
        };
        apply();
        return () => { cancelled = true; };
    }, [gameState, arAnchorId, arSurfaceOffset]);

    /** AR calibration source: center offsets computed from solved marker reports/config. */
    useEffect(() => {
        if (gameState !== 'AR_MODE') return;
        let cancelled = false;
        (async () => {
            try {
                let byId: Record<number, number[]> = {};
                let centerFromSource: MarkerOffset | undefined;

                try {
                    // Primary: public/surface_pair_report.json — marker poseMatrix + center_1346_m (meters, Three.js Y-up).
                    const sr = await fetch(`${import.meta.env.BASE_URL}surface_pair_report.json?ts=${Date.now()}`);
                    if (sr.ok) {
                        const report = await sr.json();
                        const parsed = parsePosesFromReport(report);
                        if (parsed) {
                            byId = parsed.byId;
                            centerFromSource = parsed.center;
                        }
                    }
                } catch {
                    /* optional file */
                }

                try {
                    if (Object.keys(byId).length === 0) {
                        const rr = await fetch(`${import.meta.env.BASE_URL}table_rotation_report.json?ts=${Date.now()}`);
                        if (rr.ok) {
                            const report = await rr.json();
                            const parsed = parsePosesFromReport(report);
                            if (parsed) {
                                byId = parsed.byId;
                                centerFromSource = parsed.center;
                            }
                        }
                    }
                } catch {
                    /* optional file */
                }

                if (Object.keys(byId).length === 0) {
                    const res = await fetch(`${import.meta.env.BASE_URL}config.json`);
                    const json = await res.json();
                    byId = parsePosesFromConfig(json);
                }

                const tablePoses = TABLE_MARKER_IDS.map((id) => byId[id]).filter(Boolean) as number[][];
                if (tablePoses.length === 0) {
                    if (!cancelled) setCenterOffsetsById({});
                    return;
                }

                const tablePts = tablePoses.map((pose) => translationFromPose(pose));
                const centerGlobal: MarkerOffset =
                    centerFromSource ?? {
                        x: tablePts.reduce((s, p) => s + p.x, 0) / tablePts.length,
                        y: tablePts.reduce((s, p) => s + p.y, 0) / tablePts.length,
                        z: tablePts.reduce((s, p) => s + p.z, 0) / tablePts.length,
                    };

                const next: Record<number, MarkerOffset> = {};
                for (const id of ALL_MARKER_IDS) {
                    const pose = byId[id];
                    if (!pose) continue;
                    const offM = centerOffsetInMarkerLocalFromPose(pose, centerGlobal);
                    next[id] = {
                        x: offM.x / MARKER_SIZE_METERS,
                        y: offM.y / MARKER_SIZE_METERS,
                        z: offM.z / MARKER_SIZE_METERS,
                    };
                }
                if (!cancelled) {
                    setCenterOffsetsById(next);
                }
            } catch (e) {
                console.warn('[AR] calibration load failed:', e);
                if (!cancelled) setCenterOffsetsById({});
            }
        })();
        return () => {
            cancelled = true;
        };
    }, [gameState]);

    /** AR marker visibility tracking via AR.js markerFound/markerLost events. */
    useEffect(() => {
        if (gameState !== 'AR_MODE') return;
        const els = ALL_MARKER_IDS.map((id) => document.querySelector(`a-marker[type="barcode"][value="${id}"]`));
        const onFound = (id: number) => () => {
            console.log(`[AR] markerFound id=${id}`);
            setArVisibleIds((prev) => new Set(prev).add(id));
        };
        const onLost = (id: number) => () => {
            console.log(`[AR] markerLost id=${id}`);
            setArVisibleIds((prev) => {
                const next = new Set(prev);
                next.delete(id);
                return next;
            });
        };
        const cleanups: Array<() => void> = [];
        for (let i = 0; i < ALL_MARKER_IDS.length; i++) {
            const id = ALL_MARKER_IDS[i];
            const el = els[i] as any;
            if (!el) continue;
            const f = onFound(id);
            const l = onLost(id);
            el.addEventListener('markerFound', f);
            el.addEventListener('markerLost', l);
            cleanups.push(() => {
                el.removeEventListener('markerFound', f);
                el.removeEventListener('markerLost', l);
            });
        }
        return () => cleanups.forEach((fn) => fn());
    }, [gameState]);

    /** Poll-based fallback for marker visibility + drives the scan prompt + grace period for tracking jitter. */
    useEffect(() => {
        if (gameState !== 'AR_MODE') return;
        const interval = window.setInterval(() => {
            const now = Date.now();
            const next = new Set<number>();
            for (const id of ALL_MARKER_IDS) {
                const el = document.querySelector(`a-marker[type="barcode"][value="${id}"]`) as any;
                const isVisible = Boolean(el?.object3D?.visible);
                if (isVisible) {
                    arLastSeenMsRef.current[id] = now;
                    next.add(id);
                    continue;
                }
                const lastSeen = arLastSeenMsRef.current[id] ?? 0;
                if (now - lastSeen <= markerLostGraceMs) next.add(id);
            }
            setArVisibleIds((prev) => (setsEqual(prev, next) ? prev : next));
            setScanPrompt(next.size === 0);
        }, 120);
        return () => window.clearInterval(interval);
    }, [gameState]);

    /** Anchor selection: prefer marker 4 if visible, otherwise debounce-hold on another. */
    useEffect(() => {
        if (gameState !== 'AR_MODE') return;
        if (arAnchorHoldTimeoutRef.current !== null) {
            window.clearTimeout(arAnchorHoldTimeoutRef.current);
            arAnchorHoldTimeoutRef.current = null;
        }
        if (arVisibleIds.has(4)) {
            setArAnchorId(4);
            return;
        }
        if (arAnchorId !== null && arVisibleIds.has(arAnchorId)) return;
        const next = ALL_MARKER_IDS.find((id) => arVisibleIds.has(id)) ?? null;
        arAnchorHoldTimeoutRef.current = window.setTimeout(() => {
            setArAnchorId(next);
            arAnchorHoldTimeoutRef.current = null;
        }, anchorSwitchDebounceMs);
    }, [gameState, arVisibleIds, arAnchorId]);

    const activeAnchorId = arAnchorId;

    /** Flat AR: while “Map your play zone” is up, hide HUD. After READY: HUD → field (mesh + collectibles) → rover. */
    const arMapZoneActive = gameState === 'AR_MODE' && flatSurfaceMode && arSetupPhase !== 'READY';
    const arFlatHudChromeVisible =
        gameState !== 'AR_MODE'
        || !flatSurfaceMode
        || (!arMapZoneActive && (arFlatRevealPhase === 'HUD' || arFlatRevealPhase === 'FIELD' || arFlatRevealPhase === 'ROVER'));
    const arFlatFieldEntitiesVisible =
        gameState !== 'AR_MODE'
        || !flatSurfaceMode
        || (!arMapZoneActive && (arFlatRevealPhase === 'FIELD' || arFlatRevealPhase === 'ROVER'));

    return (
        <div className="ar-container">
            {gameState === 'MENU' && (
                <div id="start-screen">
                    {/* Modern Star Field */}
                    <div style={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        width: '100%',
                        height: '100%',
                        overflow: 'hidden',
                        pointerEvents: 'none'
                    }}>
                        {/* Glowing orbs */}
                        {[
                            { top: '12%', left: '18%', size: 6, blur: 15, color: 'rgba(0, 212, 255, 0.8)', delay: 0 },
                            { top: '28%', left: '82%', size: 8, blur: 20, color: 'rgba(123, 44, 191, 0.7)', delay: 0.5 },
                            { top: '58%', left: '12%', size: 5, blur: 12, color: 'rgba(255, 255, 255, 0.9)', delay: 1 },
                            { top: '78%', left: '72%', size: 7, blur: 18, color: 'rgba(0, 212, 255, 0.6)', delay: 1.5 },
                            { top: '22%', left: '48%', size: 4, blur: 10, color: 'rgba(255, 255, 255, 0.8)', delay: 0.8 },
                            { top: '88%', left: '38%', size: 6, blur: 16, color: 'rgba(123, 44, 191, 0.6)', delay: 1.2 },
                            { top: '8%', left: '88%', size: 5, blur: 14, color: 'rgba(255, 255, 255, 0.7)', delay: 0.3 },
                            { top: '48%', left: '6%', size: 4, blur: 11, color: 'rgba(0, 212, 255, 0.7)', delay: 1.8 },
                            { top: '35%', left: '62%', size: 3, blur: 8, color: 'rgba(255, 255, 255, 0.6)', delay: 0.4 },
                            { top: '65%', left: '88%', size: 4, blur: 10, color: 'rgba(123, 44, 191, 0.5)', delay: 1.1 },
                            { top: '82%', left: '22%', size: 3, blur: 9, color: 'rgba(255, 255, 255, 0.7)', delay: 0.7 },
                            { top: '15%', left: '38%', size: 5, blur: 13, color: 'rgba(0, 212, 255, 0.6)', delay: 1.4 },
                            { top: '42%', left: '75%', size: 4, blur: 11, color: 'rgba(255, 255, 255, 0.8)', delay: 0.9 },
                            { top: '72%', left: '55%', size: 6, blur: 15, color: 'rgba(123, 44, 191, 0.7)', delay: 1.6 },
                            { top: '5%', left: '65%', size: 3, blur: 8, color: 'rgba(255, 255, 255, 0.6)', delay: 0.2 },
                            { top: '92%', left: '58%', size: 4, blur: 10, color: 'rgba(0, 212, 255, 0.5)', delay: 1.3 },
                        ].map((star, i) => (
                            <div
                                key={`star-${i}`}
                                style={{
                                    position: 'absolute',
                                    top: star.top,
                                    left: star.left,
                                    width: `${star.size}px`,
                                    height: `${star.size}px`,
                                    borderRadius: '50%',
                                    background: star.color,
                                    boxShadow: `0 0 ${star.blur}px ${star.color}, 0 0 ${star.blur * 2}px ${star.color}`,
                                    animation: `twinkle ${2.5 + Math.random() * 2}s ease-in-out infinite`,
                                    animationDelay: `${star.delay}s`,
                                }}
                            />
                        ))}
                    </div>

                    <div className="mission-badge">
                        <div className="badge-label">NASA Capstone Project</div>
                    </div>
                    <h1>Psyche</h1>
                    <p className="subtitle">Explore • Navigate • Discover</p>
                    <div className="button-container">
                        <button id="play-button" ref={playBtnRef} onClick={() => { setShowDifficulty(true); }} disabled={!meshLoaded}>
                            {meshLoaded ? 'Launch Mission' : 'Loading...'}
                        </button>
                        <button id="start-button" ref={arBtnRef} onClick={() => { if (meshLoaded) void handleStart('ar', 'easy'); }} disabled={!meshLoaded}>
                            {meshLoaded ? 'AR Experience' : 'Loading...'}
                        </button>
                        <button id="credits-button" ref={creditsBtnRef} onClick={() => setShowCredits(true)}>Credits</button>
                    </div>
                    <div className={`difficulty-overlay ${showDifficulty ? 'open' : 'closed'}`} onClick={() => setShowDifficulty(false)}>
                        <div className="difficulty-modal" onClick={(e) => e.stopPropagation()} role="dialog" aria-modal="true" aria-hidden={!showDifficulty}>
                            <h2 className="difficulty-title">Select Difficulty</h2>
                            <p className="difficulty-sub">Choose how challenging the mission will be.</p>

                            <div className="difficulty-buttons" onKeyDown={(e) => {
                                // Trap Tab navigation between the three difficulty buttons
                                if (e.key === 'Tab') {
                                    e.preventDefault();
                                    const refs = diffBtnRefs;
                                    const focusedIndex = refs.findIndex(r => r.current === document.activeElement);
                                    const dir = e.shiftKey ? -1 : 1;
                                    let next = focusedIndex + dir;
                                    if (next < 0) next = refs.length - 1;
                                    if (next >= refs.length) next = 0;
                                    refs[next].current?.focus();
                                }
                            }}>
                                <button ref={diffBtnRefs[0]} className="difficulty-btn" onClick={() => { setShowDifficulty(false); void handleStart('web_game', 'easy'); }}>Story</button>
                                <button ref={diffBtnRefs[1]} className="difficulty-btn" onClick={() => { setShowDifficulty(false); void handleStart('web_game', 'normal'); }}>Standard</button>
                                <button ref={diffBtnRefs[2]} className="difficulty-btn" onClick={() => { setShowDifficulty(false); void handleStart('web_game', 'hard'); }}>Challenge</button>
                            </div>
                        </div>
                    </div>
                    <div className={`credits-overlay ${showCredits ? 'open' : 'closed'}`} onClick={() => setShowCredits(false)}>
                        <div className="credits-modal" onClick={(e) => e.stopPropagation()} role="dialog" aria-modal="true" aria-hidden={!showCredits}>
                            <h2 className="credits-title">Credits</h2>

                            <section className="credits-section">
                                <h3 className="credits-section-heading">Creators</h3>
                                <ul className="credits-list">
                                    <li>Matthew Andrews — Systems Programmer</li>
                                    <li>Brian Devaney — Technical Director</li>
                                    <li>Methsiri Faris — AR Engineer</li>
                                    <li>Evelyn Giordano — Technical Artist</li>
                                    <li>Nathaniel Wilson — Gameplay Programmer</li>
                                </ul>
                            </section>

                            <section className="credits-section">
                                <h3 className="credits-section-heading">Sponsors</h3>
                                <ul className="credits-list">
                                    <li>NASA Psyche Mission</li>
                                    <li>Cassie Bowman — Arizona State University </li>
                                    <li>Alejandro Gomez — University of Arkansas</li>
                                </ul>
                            </section>

                            <section className="credits-section">
                                <h3 className="credits-section-heading">Citations</h3>
                                <ul className="credits-list">
                                    <li>NASA JPL Psyche Press Kit: https://www.jpl.nasa.gov/press-kits/psyche/</li>
                                    <li>Psyche Mission FAQ: https://psyche.ssl.berkeley.edu/mission/faq/</li>
                                </ul>
                            </section>

                            <section className="credits-section">
                                <h3 className="credits-section-heading">Disclaimer</h3>
                                <p className="credits-disclaimer">
                                    This work was created in partial fulfillment of University of Arkansas Capstone Course "CSCE 49603 - Capstone II". The work is a result of the Psyche Student Collaborations component of NASA's Psyche Mission (https://psyche.ssl.berkeley.edu)
                                    "Psyche: A Journey to a Metal World" [Contract number NNM16AA09C] is part of the NASA Discovery Program mission to solar system targets. Trade names and trademarks of ASU and NASA are used in this work for identification only.
                                    Their usage does not constitute an official endorsement, either expressed or implied, by Arizona State University or National Aeronautics and Space Administration. The content is solely the responsibility of the authors and does not necessarily represent the official views of ASU or NASA.
                                    <br />
                                    The use of a rover on Psyche is not mission accurate and is included for gameplay purposes only.
                                </p>
                            </section>

                            <button className="credits-close-btn" onClick={() => setShowCredits(false)}>Close</button>
                        </div>
                    </div>
                </div>
            )}

            {gameState === 'AR_MODE' && (
                <>
                    {/* AR Scene with Camera Access — marker-anchored, calibrated asteroid world. */}
                    <div style={{ position: 'fixed', top: 0, left: 0, width: '100vw', height: '100dvh', zIndex: 0, overflow: 'hidden' }}>
                        <a-scene
                            embedded
                            style={{ position: 'fixed', top: 0, left: 0, width: '100vw', height: '100dvh' }}
                            arjs="sourceType: webcam; detectionMode: mono_and_matrix; matrixCodeType: 3x3_HAMMING63; patternRatio: 0.52;"
                            vr-mode-ui="enabled: false"
                            renderer="logarithmicDepthBuffer: true; antialias: true; maxCanvasWidth: 1280; maxCanvasHeight: 1280;"
                        >
                            <a-camera position="0 0 0" look-controls="enabled: false"></a-camera>

                            {/* Lighting so the glb samples/asteroid are readable in AR */}
                            <a-light type="ambient" color="#FFFFFF" intensity="0.9"></a-light>
                            <a-light type="directional" color="#FFFFFF" intensity="0.8" position="3 5 4"></a-light>

                            {[...TABLE_MARKER_IDS, ...SURFACE_MARKER_IDS].map((id) => {
                                const c = centerOffsetsById[id] ?? { x: 0, y: 0, z: 0 };
                                return (
                                    <a-marker
                                        key={id}
                                        type="barcode"
                                        value={id}
                                        size={MARKER_SIZE_METERS}
                                        smooth="true"
                                        smoothCount="18"
                                        smoothTolerance="0.008"
                                        smoothThreshold="4"
                                    >
                                        {/*
                                          * Calibration reference cube. Direct child of <a-marker>, so it lives in the
                                          * marker's raw local frame — NO shared transform with the asteroid / rover / samples.
                                          * Purely a visual probe: if you see a red cube on the physical marker, AR.js is
                                          * tracking and your calibration offsets are sensible. Toggle via showCalibrationCube.
                                          */}
                                        {showCalibrationCube && (
                                            <a-box
                                                position={`${markerOverlayShiftX} ${markerPlaneOffset + markerOverlayDepth / 2} ${markerOverlayShiftZ}`}
                                                rotation="-90 0 0"
                                                width={markerOverlayWidth}
                                                height={markerOverlayDepth}
                                                depth={markerOverlayHeight}
                                                material="color: #ff0000; shader: flat; side: double; transparent: true; opacity: 0.55"
                                            />
                                        )}

                                        {activeAnchorId === id && (
                                            <a-entity position={`${c.x} ${c.y} ${c.z}`}>
                                                {/*
                                                 * Virtual asteroid GLB: full calibration stack (lift / rotation / scale).
                                                 * Kept in its own nested entity so the flat-play frame below can stay
                                                 * scale-only — critical because tap raycasts land on the play-root's
                                                 * local XZ plane and any rotation there would skew the pick math.
                                                 */}
                                                {showArAsteroid && (
                                                    <a-entity
                                                        position={`0 ${modelLift} ${modelBack}`}
                                                        rotation={`${modelPitchOffsetDeg} ${modelYawOffsetDeg} ${modelRollOffsetDeg}`}
                                                        scale={`${modelScaleX} ${modelScaleY} ${modelScaleZ}`}
                                                    >
                                                        <a-entity position={arAsteroidGltfPosition} scale={arAsteroidGltfScale}>
                                                            <a-gltf-model src={arAsteroidModelSrc} />
                                                        </a-entity>
                                                    </a-entity>
                                                )}
                                                <a-entity
                                                    id="ar-play-root"
                                                    scale={`${modelScaleX} ${modelScaleY} ${modelScaleZ}`}
                                                >
                                                    {/*
                                                     * Flat-surface walking disk. The rover, samples, and obstacles all
                                                     * live on this disk. In setup phase we also render tap pins (center,
                                                     * edge preview) so the user can see where they've tapped in 3D AR.
                                                     */}
                                                    {flatSurfaceMode && arSetupPhase !== 'AWAIT_CENTER' && (
                                                        // Center pin — glowing dot at the tapped disk center.
                                                        <a-entity position={`${flatSurfaceOffsetX} ${flatSurfaceHeight + 0.02} ${flatSurfaceOffsetZ}`}>
                                                            <a-ring
                                                                radius-inner="0.05"
                                                                radius-outer="0.12"
                                                                rotation="-90 0 0"
                                                                color="#ffd86b"
                                                                material="shader: flat; transparent: true; opacity: 0.9; side: double"
                                                            />
                                                            <a-sphere
                                                                radius="0.05"
                                                                color="#ffd86b"
                                                                material="shader: flat; emissive: #ffd86b; emissiveIntensity: 1; transparent: true; opacity: 0.95"
                                                                animation="property: scale; from: 1 1 1; to: 1.6 1.6 1.6; loop: true; dir: alternate; dur: 700; easing: easeInOutSine"
                                                            />
                                                        </a-entity>
                                                    )}
                                                    {/*
                                                     * CAPTURED TERRAIN MESH — displaced CircleGeometry built from the OpenCV
                                                     * heightmap + textured with the snapshot we took on the second tap.
                                                     * Rendered only after tap-to-map completes and a terrain token exists.
                                                     * The rover + samples + obstacles already have their Y values snapped
                                                     * to this same heightmap (see spawn + movement code), so visuals and
                                                     * physics align.
                                                     */}
                                                    {/* Visual only: heightmap lives in terrainRef; physics ignores hideTerrainSurface. */}
                                                    {flatSurfaceMode && arSetupPhase === 'READY' && arFlatFieldEntitiesVisible && terrainToken && !hideTerrainSurface && (
                                                        <a-entity
                                                            position={`${flatSurfaceOffsetX} ${flatSurfaceHeight} ${flatSurfaceOffsetZ}`}
                                                            {...{ 'heightmap-terrain': `token: ${terrainToken}; segments: 112; opacity: 0.72` }}
                                                        />
                                                    )}

                                                    {/* Samples — GLB models matching the web game, scaled down for the marker-anchored world. */}
                                                    {samples.map((s) => (
                                                        <a-entity key={`ar-${s.id}`} position={`${s.x} ${s.y} ${s.z}`} rotation={s.rotation} visible={arFlatFieldEntitiesVisible ? 'true' : 'false'}>
                                                            <a-gltf-model src={`./models/${s.model}.glb`} scale={arSampleScaleStr} />
                                                        </a-entity>
                                                    ))}

                                                    {/* Obstacles: collision + energy still use `obstacles`; red markers removed per user request. */}

                                                    {/* Nearest-sample indicator arrow (tangent-plane orbit). */}
                                                    <a-entity id="sample-arrow" visible="false">
                                                        <a-entity animation="property: scale; from: 1 1 1; to: 1.35 1.35 1.35; loop: true; dir: alternate; dur: 500; easing: easeInOutSine">
                                                            <a-cone
                                                                height={arArrowConeHeight}
                                                                radius-bottom={arArrowConeRadiusBottom}
                                                                radius-top="0"
                                                                color="#FFD700"
                                                                position={`0 ${arArrowConeY} 0`}
                                                                material="emissive: #FFD700; emissiveIntensity: 0.55; transparent: true; opacity: 0.95"
                                                            />
                                                            <a-cylinder
                                                                radius={arArrowCylRadius}
                                                                height={arArrowCylHeight}
                                                                color="#FFD700"
                                                                position={`0 ${arArrowCylY} 0`}
                                                                material="emissive: #FFD700; emissiveIntensity: 0.35; transparent: true; opacity: 0.8"
                                                            />
                                                        </a-entity>
                                                    </a-entity>

                                                    {/* Rover — same GLB as web (`craft_racer.glb`); parent scale compensates for non-uniform marker transform.
                                                        In flat mode we hide it until tap-to-map completes so the player sees the gray surface first. */}
                                                    <a-entity
                                                        id="ar-rover"
                                                        position="0 0 0"
                                                        rotation="0 0 0"
                                                        scale={arRoverScaleStr}
                                                        visible={roverReady && (!flatSurfaceMode || (arSetupPhase === 'READY' && arFlatRevealPhase === 'ROVER')) ? 'true' : 'false'}
                                                    >
                                                        {/* Debug sphere — bright unlit green, always-on-top, co-located with rover pivot. */}
                                                        {showArRoverDebugSphere && (
                                                            <a-sphere
                                                                radius="0.35"
                                                                color="#00ff6a"
                                                                material="shader: flat; transparent: true; opacity: 0.55; depthTest: false"
                                                                position="0 0 0"
                                                            />
                                                        )}
                                                        <a-gltf-model
                                                            src="./models/craft_racer.glb"
                                                            scale="0.2 0.2 0.2"
                                                        />
                                                    </a-entity>
                                                </a-entity>
                                            </a-entity>
                                        )}
                                    </a-marker>
                                );
                            })}
                        </a-scene>
                    </div>

                    <div id="ui-overlay" style={{ display: 'block' }}>
                        {scanPrompt && !(flatSurfaceMode && arSetupPhase !== 'READY') && arSetupPhase === 'AWAIT_CENTER' && arAnchorId === null && (
                            <div id="scan-prompt">
                                Point camera at AR marker
                            </div>
                        )}

                        {/*
                          * TAP-TO-MAP SETUP OVERLAY — creative, phase-driven prompts that replace the
                          * slider calibration for first-time users. Each phase is one clear action:
                          *   AWAIT_CENTER: "Tap the center of the gray face"
                          *   AWAIT_EDGE:   "Tap the edge to size your play zone"
                          *   READY:        hidden (game is playing)
                          */}
                        {gameState === 'AR_MODE' && flatSurfaceMode && arSetupPhase !== 'READY' && (
                            <div
                                className="ar-setup-overlay"
                                style={{
                                    position: 'fixed',
                                    left: '50%',
                                    top: 16,
                                    transform: 'translateX(-50%)',
                                    zIndex: 1002,
                                    padding: '14px 22px',
                                    borderRadius: 14,
                                    background: 'linear-gradient(135deg, rgba(16, 26, 42, 0.92), rgba(22, 44, 64, 0.92))',
                                    border: '1px solid rgba(123, 255, 178, 0.45)',
                                    boxShadow: '0 10px 30px rgba(0,0,0,0.45), 0 0 24px rgba(123,255,178,0.18) inset',
                                    color: '#E6F2FF',
                                    fontFamily: 'inherit',
                                    textAlign: 'center',
                                    minWidth: 280,
                                    maxWidth: '90vw',
                                    backdropFilter: 'blur(8px)',
                                    pointerEvents: 'auto',
                                }}
                            >
                                <div style={{
                                    fontSize: 10,
                                    fontWeight: 800,
                                    letterSpacing: 2,
                                    color: '#7bffb2',
                                    marginBottom: 6,
                                    textTransform: 'uppercase',
                                }}>
                                    MAP YOUR PLAY ZONE · STEP {arSetupPhase === 'AWAIT_CENTER' ? '1' : '2'} OF 2
                                </div>
                                <div style={{ fontSize: 17, fontWeight: 700, marginBottom: 4 }}>
                                    {arAnchorId === null
                                        ? 'Looking for marker…'
                                        : arSetupPhase === 'AWAIT_CENTER'
                                            ? 'Tap the center of the gray face'
                                            : 'Tap the edge to size your play zone'}
                                </div>
                                <div style={{ fontSize: 12, opacity: 0.8, lineHeight: 1.35 }}>
                                    {arAnchorId === null
                                        ? 'Keep a marker in view, then tap directly on the printed asteroid.'
                                        : arSetupPhase === 'AWAIT_CENTER'
                                            ? 'This is where the rover will land.'
                                            : 'Move your finger from the glowing pin — the farther you tap, the bigger your territory.'}
                                </div>
                                <div style={{
                                    display: 'flex',
                                    gap: 8,
                                    justifyContent: 'center',
                                    marginTop: 12,
                                }}>
                                    {arSetupPhase === 'AWAIT_EDGE' && (
                                        <button
                                            type="button"
                                            onClick={() => {
                                                setArSetupPhase('AWAIT_CENTER');
                                                updateArCalibration({ flatSurfaceOffsetX: 0, flatSurfaceOffsetZ: 0 });
                                            }}
                                            style={{
                                                padding: '6px 12px',
                                                borderRadius: 8,
                                                border: '1px solid rgba(255,255,255,0.25)',
                                                background: 'rgba(255,255,255,0.06)',
                                                color: '#E6F2FF',
                                                fontSize: 11,
                                                fontWeight: 700,
                                                cursor: 'pointer',
                                            }}
                                        >
                                            ← Redo center
                                        </button>
                                    )}
                                    {arAnchorId !== null && (
                                        <button
                                            type="button"
                                            onClick={() => {
                                                // Skip path — drop a default disk centered on the marker.
                                                updateArCalibration({
                                                    flatSurfaceOffsetX: 0,
                                                    flatSurfaceOffsetZ: 0,
                                                    flatSurfaceRadius: (PHYSICAL_ASTEROID_BBOX_M.x / 2) / MARKER_SIZE_METERS,
                                                });
                                                setArSetupPhase('READY');
                                            }}
                                            style={{
                                                padding: '6px 12px',
                                                borderRadius: 8,
                                                border: '1px solid rgba(123,255,178,0.3)',
                                                background: 'rgba(24, 52, 40, 0.7)',
                                                color: '#7bffb2',
                                                fontSize: 11,
                                                fontWeight: 700,
                                                cursor: 'pointer',
                                            }}
                                        >
                                            Skip (use default zone)
                                        </button>
                                    )}
                                </div>
                            </div>
                        )}

                        {/*
                          * Post-setup HUD chip — Retake / Remap only (zone size + terrain badge removed).
                          */}
                        {gameState === 'AR_MODE' && flatSurfaceMode && arSetupPhase === 'READY' && arFlatHudChromeVisible && (
                            <div
                                className="ar-setup-overlay"
                                style={{
                                    position: 'fixed',
                                    left: 12,
                                    bottom: 12,
                                    zIndex: 1001,
                                    padding: '6px 10px',
                                    borderRadius: 10,
                                    background: 'rgba(12, 20, 32, 0.7)',
                                    border: '1px solid rgba(123,255,178,0.3)',
                                    color: '#E6F2FF',
                                    fontSize: 11,
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: 10,
                                    pointerEvents: 'auto',
                                }}
                            >
                                <button
                                    type="button"
                                    onClick={() => {
                                        void (async () => {
                                            try {
                                                const cv = (window as any).cv;
                                                const video = document.querySelector('video') as HTMLVideoElement | null;
                                                if (cv && video && video.videoWidth > 0) {
                                                    const centerPx = {
                                                        x: video.videoWidth / 2,
                                                        y: video.videoHeight / 2,
                                                    };
                                                    const rPx = Math.min(video.videoWidth, video.videoHeight) * 0.4;
                                                    const hm = buildHeightmapFromFrame(cv, video, centerPx, rPx, 96);
                                                    if (hm) {
                                                        await augmentHeightmapWithDepthFromMotion(cv, video, hm);
                                                        terrainRef.current = hm;
                                                        const token = `terrain-${Date.now()}`;
                                                        _terrainPayloadRegistry.set(token, { hm, radius: flatSurfaceRadius });
                                                        setTerrainToken(token);
                                                        setArDepthHeatmapUrl(hm.depthDebugUrl ?? null);
                                                        roverPosRef.current = null;
                                                    }
                                                }
                                            } catch (err) {
                                                console.warn('[AR][terrain] retake failed:', err);
                                            }
                                        })();
                                    }}
                                    style={{
                                        padding: '4px 10px',
                                        borderRadius: 6,
                                        border: '1px solid rgba(255,255,255,0.25)',
                                        background: 'rgba(255,255,255,0.06)',
                                        color: '#E6F2FF',
                                        fontSize: 10,
                                        fontWeight: 700,
                                        cursor: 'pointer',
                                    }}
                                >
                                    Retake
                                </button>
                                <button
                                    type="button"
                                    onClick={() => {
                                        roverPosRef.current = null;
                                        terrainRef.current = null;
                                        setTerrainToken(null);
                                        setArDepthHeatmapUrl(null);
                                        setArSetupPhase('AWAIT_CENTER');
                                    }}
                                    style={{
                                        padding: '4px 10px',
                                        borderRadius: 6,
                                        border: '1px solid rgba(255,255,255,0.25)',
                                        background: 'rgba(255,255,255,0.06)',
                                        color: '#E6F2FF',
                                        fontSize: 10,
                                        fontWeight: 700,
                                        cursor: 'pointer',
                                    }}
                                >
                                    Remap
                                </button>
                            </div>
                        )}

                        {gameState === 'AR_MODE' && flatSurfaceMode && arDepthHeatmapUrl && arFlatHudChromeVisible && (
                            <div
                                style={{
                                    position: 'fixed',
                                    right: 14,
                                    bottom: 218,
                                    zIndex: 1000,
                                    pointerEvents: 'none',
                                    textAlign: 'right',
                                }}
                            >
                                <div style={{ fontSize: 9, fontWeight: 700, letterSpacing: 0.4, color: '#b8e8ff', opacity: 0.9, marginBottom: 4 }}>
                                    Depth (motion)
                                </div>
                                <img
                                    src={arDepthHeatmapUrl}
                                    alt=""
                                    width={72}
                                    height={72}
                                    style={{
                                        borderRadius: 8,
                                        border: '1px solid rgba(123,255,178,0.35)',
                                        boxShadow: '0 6px 18px rgba(0,0,0,0.35)',
                                        display: 'block',
                                    }}
                                />
                            </div>
                        )}

                        <div
                            style={{
                                visibility: arFlatHudChromeVisible ? 'visible' : 'hidden',
                                pointerEvents: arFlatHudChromeVisible ? 'auto' : 'none',
                            }}
                        >
                            <div id="score-display">
                                SCORE <span id="score">{score}</span>
                            </div>

                            <div className="mode-ui">
                                {modeCfg.energyEnabled && <div className="energy-display">ENERGY <div className="energy-bar"><div style={{ width: `${(energy / MAX_ENERGY) * 100}%` }} /></div></div>}
                                <div className="samples-display">SAMPLES <span style={{ color: '#7bffb2', fontWeight: 800 }}>{samplesCollected}</span> / {modeCfg.spawnSamples}</div>
                            </div>
                        </div>

                        {false && (
                            <div
                                style={{
                                    position: 'fixed',
                                    top: 52,
                                    right: 12,
                                    zIndex: 1001,
                                    width: 300,
                                    maxHeight: 'calc(100vh - 80px)',
                                    overflowY: 'auto',
                                    padding: 14,
                                    borderRadius: 12,
                                    border: '1px solid rgba(255,255,255,0.18)',
                                    background: 'rgba(10, 14, 24, 0.85)',
                                    color: '#E6F2FF',
                                    fontSize: 12,
                                    fontFamily: 'system-ui, sans-serif',
                                    backdropFilter: 'blur(8px)',
                                    boxShadow: '0 8px 28px rgba(0,0,0,0.45)',
                                    pointerEvents: 'auto',
                                }}
                                onClick={(e) => e.stopPropagation()}
                            >
                                <div style={{ fontWeight: 800, letterSpacing: 0.6, marginBottom: 8 }}>AR ASTEROID ALIGNMENT</div>
                                <div style={{ fontSize: 10, opacity: 0.75, marginBottom: 10, lineHeight: 1.45 }}>
                                    Physical print {Math.round(PHYSICAL_ASTEROID_BBOX_M.x * 1000)}×{Math.round(PHYSICAL_ASTEROID_BBOX_M.y * 1000)}×{Math.round(PHYSICAL_ASTEROID_BBOX_M.z * 1000)} mm;
                                    centers from <code style={{ fontSize: 9 }}>surface_pair_report.json</code>.
                                    Matched uniform scale ≈{' '}
                                    <span style={{ color: '#7bffb2' }}>{AR_PHYSICAL_MATCH_UNIFORM_SCALE.toFixed(3)}</span>
                                    {' '}(marker units; 1 unit = {MARKER_SIZE_METERS * 1000} mm).
                                </div>
                                <button
                                    type="button"
                                    onClick={() => {
                                        const s = AR_PHYSICAL_MATCH_UNIFORM_SCALE;
                                        setArCalibration((prev) => ({
                                            ...prev,
                                            modelScaleX: s,
                                            modelScaleY: s,
                                            modelScaleZ: s,
                                            modelLift: (-30.15 * s) / LEGACY_AR_MODEL_SCALE_REF,
                                            modelBack: 0,
                                        }));
                                    }}
                                    style={{
                                        width: '100%',
                                        marginBottom: 12,
                                        padding: '8px 10px',
                                        borderRadius: 8,
                                        border: '1px solid rgba(123,255,178,0.35)',
                                        background: 'rgba(24, 52, 40, 0.9)',
                                        color: '#7bffb2',
                                        fontSize: 11,
                                        fontWeight: 800,
                                        cursor: 'pointer',
                                    }}
                                >
                                    Match physical print (610×524×432 mm)
                                </button>
                                <div style={{
                                    display: 'flex',
                                    flexDirection: 'column',
                                    gap: 6,
                                    marginBottom: 12,
                                    padding: 8,
                                    borderRadius: 8,
                                    background: 'rgba(255,255,255,0.05)',
                                    border: '1px solid rgba(255,255,255,0.12)',
                                }}>
                                    <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: 0.5, opacity: 0.85 }}>
                                        DIGITAL-TWIN RENDERING
                                    </div>
                                    <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', fontSize: 11 }}>
                                        <input
                                            type="checkbox"
                                            checked={hideVirtualAsteroid}
                                            onChange={(e) => updateArCalibration({ hideVirtualAsteroid: e.target.checked })}
                                        />
                                        <span>Hide virtual asteroid (see real rock through camera; rover + crystals + obstacles ride the invisible twin)</span>
                                    </label>
                                    <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', fontSize: 11 }}>
                                        <input
                                            type="checkbox"
                                            checked={showCalibrationCube}
                                            onChange={(e) => updateArCalibration({ showCalibrationCube: e.target.checked })}
                                        />
                                        <span>Show red 2″ marker cube (diagnostic only)</span>
                                    </label>
                                </div>
                                <div style={{
                                    display: 'flex',
                                    flexDirection: 'column',
                                    gap: 6,
                                    marginBottom: 12,
                                    padding: 8,
                                    borderRadius: 8,
                                    background: 'rgba(123,255,178,0.06)',
                                    border: '1px solid rgba(123,255,178,0.25)',
                                }}>
                                    <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: 0.5, opacity: 0.85 }}>
                                        FLAT-SURFACE ROVER (gray face of printed asteroid)
                                    </div>
                                    <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', fontSize: 11 }}>
                                        <input
                                            type="checkbox"
                                            checked={flatSurfaceMode}
                                            onChange={(e) => updateArCalibration({ flatSurfaceMode: e.target.checked })}
                                        />
                                        <span>Flat-surface mode (rover walks on a disk instead of wrapping the full sphere)</span>
                                    </label>
                                    <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', fontSize: 11 }}>
                                        <input
                                            type="checkbox"
                                            checked={showFlatDisk}
                                            onChange={(e) => updateArCalibration({ showFlatDisk: e.target.checked })}
                                        />
                                        <span>Show green disk outline (turn off for clean play)</span>
                                    </label>
                                    <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', fontSize: 11 }}>
                                        <input
                                            type="checkbox"
                                            checked={hideTerrainSurface}
                                            onChange={(e) => updateArCalibration({ hideTerrainSurface: e.target.checked })}
                                        />
                                        <span>
                                            Hide terrain mesh only (snapshot overlay). Rover tilt, height, samples, and obstacles still use the captured heightmap; turn off to draw the textured bump map on top of the print.
                                        </span>
                                    </label>
                                    <div>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                                            <span>Disk radius (marker units)</span>
                                            <span style={{ color: '#7bffb2', fontVariantNumeric: 'tabular-nums' }}>
                                                {flatSurfaceRadius.toFixed(2)} ≈ {(flatSurfaceRadius * MARKER_SIZE_METERS * 1000).toFixed(0)} mm
                                            </span>
                                        </div>
                                        <input
                                            type="range"
                                            min={1}
                                            max={20}
                                            step={0.1}
                                            value={flatSurfaceRadius}
                                            onChange={(e) => updateArCalibration({ flatSurfaceRadius: parseFloat(e.target.value) })}
                                            style={{ width: '100%', accentColor: '#7bffb2' }}
                                        />
                                    </div>
                                    <div>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                                            <span>Disk height (marker units)</span>
                                            <span style={{ color: '#7bffb2', fontVariantNumeric: 'tabular-nums' }}>{flatSurfaceHeight.toFixed(2)}</span>
                                        </div>
                                        <input
                                            type="range"
                                            min={-20}
                                            max={20}
                                            step={0.1}
                                            value={flatSurfaceHeight}
                                            onChange={(e) => updateArCalibration({ flatSurfaceHeight: parseFloat(e.target.value) })}
                                            style={{ width: '100%', accentColor: '#7bffb2' }}
                                        />
                                    </div>
                                    <GrayDiskAutoFitRow
                                        cvReady={cvReady}
                                        lastDetection={grayDetection}
                                        onApply={(detectedRadiusMarkerUnits) => {
                                            updateArCalibration({ flatSurfaceRadius: detectedRadiusMarkerUnits });
                                        }}
                                    />
                                </div>
                                <label style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10, cursor: 'pointer', fontSize: 11 }}>
                                    <input
                                        type="checkbox"
                                        checked={compensateScaleWithLift}
                                        onChange={(e) => updateArCalibration({ compensateScaleWithLift: e.target.checked })}
                                    />
                                    <span>
                                        Auto-scale with Lift (keep ~same on-screen size when you push the rock deeper on Y — e.g. scale 12 at Y −60 vs Y −90)
                                    </span>
                                </label>
                                {compensateScaleWithLift && (
                                    <div style={{ marginBottom: 10 }}>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                                            <span>Lift depth pivot</span>
                                            <span style={{ color: '#7bffb2', fontVariantNumeric: 'tabular-nums' }}>{liftDistancePivot.toFixed(1)}</span>
                                        </div>
                                        <input
                                            type="range"
                                            min={-40}
                                            max={40}
                                            step={0.5}
                                            value={liftDistancePivot}
                                            onChange={(e) => updateArCalibration({ liftDistancePivot: parseFloat(e.target.value) })}
                                            style={{ width: '100%', accentColor: '#7bffb2' }}
                                        />
                                        <div style={{ fontSize: 9, opacity: 0.65, marginTop: 2 }}>
                                            Depth uses (pivot − lift); default 0 → deeper negative lift increases scale. Nudge pivot if ratios feel off.
                                        </div>
                                    </div>
                                )}
                                <div style={{ marginBottom: 8 }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                                        <span>Lift (Y offset)</span>
                                        <span style={{ color: '#7bffb2', fontVariantNumeric: 'tabular-nums' }}>{modelLift.toFixed(2)}</span>
                                    </div>
                                    <input
                                        type="range"
                                        min={-120}
                                        max={60}
                                        step={0.05}
                                        value={modelLift}
                                        onChange={(e) => {
                                            const newLift = parseFloat(e.target.value);
                                            setArCalibration((prev) => {
                                                if (!prev.compensateScaleWithLift) {
                                                    return { ...prev, modelLift: newLift };
                                                }
                                                const dOld = liftDepthForScreenCompensation(prev.liftDistancePivot, prev.modelLift);
                                                const dNew = liftDepthForScreenCompensation(prev.liftDistancePivot, newLift);
                                                const ratio = dNew / dOld;
                                                const clampS = (n: number) => Math.min(48, Math.max(0.2, n));
                                                return {
                                                    ...prev,
                                                    modelLift: newLift,
                                                    modelScaleX: clampS(prev.modelScaleX * ratio),
                                                    modelScaleY: clampS(prev.modelScaleY * ratio),
                                                    modelScaleZ: clampS(prev.modelScaleZ * ratio),
                                                };
                                            });
                                        }}
                                        style={{ width: '100%', accentColor: '#7bffb2' }}
                                    />
                                </div>
                                {([
                                    { key: 'modelBack', label: 'Back (Z offset)', min: -30, max: 30, step: 0.05 },
                                    { key: 'modelScaleX', label: 'Scale X', min: 0.5, max: 48, step: 0.05 },
                                    { key: 'modelScaleY', label: 'Scale Y', min: 0.5, max: 48, step: 0.05 },
                                    { key: 'modelScaleZ', label: 'Scale Z', min: 0.5, max: 48, step: 0.05 },
                                    { key: 'modelYawOffsetDeg', label: 'Yaw (°)', min: -180, max: 180, step: 0.5 },
                                    { key: 'modelPitchOffsetDeg', label: 'Pitch (°)', min: -180, max: 180, step: 0.5 },
                                    { key: 'modelRollOffsetDeg', label: 'Roll (°)', min: -180, max: 180, step: 0.5 },
                                    { key: 'sampleScaleFr', label: 'Sample scale', min: 0.005, max: 0.6, step: 0.001 },
                                ] as const).map(({ key, label, min, max, step }) => (
                                    <div key={key} style={{ marginBottom: 8 }}>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                                            <span>{label}</span>
                                            <span style={{ color: '#7bffb2', fontVariantNumeric: 'tabular-nums' }}>
                                                {(arCalibration[key] as number).toFixed(step < 0.01 ? 3 : 2)}
                                            </span>
                                        </div>
                                        <input
                                            type="range"
                                            min={min}
                                            max={max}
                                            step={step}
                                            value={arCalibration[key] as number}
                                            onChange={(e) => updateArCalibration({ [key]: parseFloat(e.target.value) } as Partial<ArCalibration>)}
                                            style={{ width: '100%', accentColor: '#7bffb2' }}
                                        />
                                    </div>
                                ))}
                                <div style={{ display: 'flex', gap: 6, marginTop: 10 }}>
                                    <button
                                        type="button"
                                        onClick={() => setArCalibration({ ...AR_CALIBRATION_DEFAULTS })}
                                        style={{
                                            flex: 1,
                                            padding: '6px 10px',
                                            borderRadius: 6,
                                            border: '1px solid rgba(255,255,255,0.25)',
                                            background: 'rgba(30, 38, 56, 0.8)',
                                            color: '#E6F2FF',
                                            fontSize: 11,
                                            fontWeight: 700,
                                            cursor: 'pointer',
                                        }}
                                    >
                                        Reset
                                    </button>
                                    <button
                                        type="button"
                                        onClick={() => {
                                            const json = JSON.stringify(arCalibration, null, 2);
                                            if (navigator.clipboard) {
                                                navigator.clipboard.writeText(json).catch(() => {
                                                    console.log('[AR] calibration JSON:', json);
                                                });
                                            }
                                        }}
                                        style={{
                                            flex: 1,
                                            padding: '6px 10px',
                                            borderRadius: 6,
                                            border: '1px solid rgba(123,255,178,0.4)',
                                            background: 'rgba(28, 64, 46, 0.8)',
                                            color: '#7bffb2',
                                            fontSize: 11,
                                            fontWeight: 700,
                                            cursor: 'pointer',
                                        }}
                                    >
                                        Copy JSON
                                    </button>
                                </div>
                                <div style={{ marginTop: 10, fontSize: 10, opacity: 0.7, lineHeight: 1.4 }}>
                                    Tip: physical-size match and “same pixels when deeper” are different goals — use the checkbox for presentation AR. True screen-constant size would need camera distance (not in this heuristic). Prefer uniform X/Y/Z scale for physics/rover fidelity.
                                </div>
                            </div>
                        )}

                        {/* Mission info popup — same overlay/modal as web so CSS (max width, scroll, z-index) applies in AR. */}
                        <div
                            className={`sample-overlay ${waypointPopup ? 'open' : 'closed'}`}
                            onClick={() => setWaypointPopup(null)}
                            role="presentation"
                        >
                            <div
                                className="sample-modal"
                                onClick={(e) => e.stopPropagation()}
                                role="dialog"
                                aria-modal="true"
                                aria-hidden={!waypointPopup}
                            >
                                {waypointPopup?.image && (
                                    <div className="sample-image-panel">
                                        <img src={waypointPopup.image} alt="" />
                                    </div>
                                )}
                                <div className="sample-text-panel">
                                    <h2 className="sample-title">{waypointPopup?.title}</h2>
                                    {waypointPopup?.body && (
                                        <p className="sample-body">{waypointPopup.body}</p>
                                    )}
                                    <button
                                        type="button"
                                        ref={sampleContinueBtnRef}
                                        className="sample-continue-btn"
                                        onClick={() => setWaypointPopup(null)}
                                    >
                                        Continue
                                    </button>
                                    <div className="sample-hint">Tap outside or Continue to close</div>
                                </div>
                            </div>
                        </div>

                        {/* End screen */}
                        {showEndScreen && (
                            <div className="end-overlay" role="dialog" aria-modal="true">
                                <div className="end-modal">
                                    <h2 className="end-title">
                                        {endReason === 'complete' ? 'Mission Complete!' : 'Out of Energy'}
                                    </h2>
                                    <p className="end-subtitle">
                                        {endReason === 'complete'
                                            ? 'All samples have been recovered from the surface of Psyche.'
                                            : "Your rover's battery has been depleted. Mission over."}
                                    </p>
                                    <div className="end-stats">
                                        <div className="end-stat">
                                            <span className="end-stat-label">Samples Collected</span>
                                            <span className="end-stat-value">{samplesCollected} / {modeCfg.spawnSamples}</span>
                                        </div>
                                        {energyBonus > 0 && (
                                            <div className="end-stat">
                                                <span className="end-stat-label">Energy Bonus</span>
                                                <span className="end-stat-value">+{energyBonus}</span>
                                            </div>
                                        )}
                                        <div className="end-stat">
                                            <span className="end-stat-label">Final Score</span>
                                            <span className="end-stat-value">{score}</span>
                                        </div>
                                    </div>
                                    <button className="end-menu-btn" onClick={returnToMenu}>
                                        Return to Main Menu
                                    </button>
                                </div>
                            </div>
                        )}

                        {/* Intro briefing — only shown when launched with a difficulty. */}
                        {showIntroPopup && (
                            <div
                                className="intro-overlay"
                                onClick={() => { if (introPopupCanClose) closeIntroPopup(); }}
                                role="dialog"
                                aria-modal="true"
                            >
                                <div className="intro-modal">
                                    <h2 className="intro-title">{INTRO_CONTENT[difficulty].welcome}</h2>
                                    <div className="intro-section">
                                        <h3 className="intro-section-heading">Controls</h3>
                                        <div className="intro-controls-grid">
                                            <span className="key-hint">W / A / S / D</span><span>Move rover</span>
                                            <span className="key-hint">Arrow Keys</span><span>Move rover</span>
                                            <span className="key-hint">D-pad</span><span>Move rover (mobile/touch)</span>
                                        </div>
                                    </div>
                                    <div className="intro-section">
                                        <p className="intro-description">{INTRO_CONTENT[difficulty].description}</p>
                                        <p className="intro-description" style={{ marginTop: '0.75rem', opacity: 0.9 }}>
                                            In AR, point the camera at your printed markers until the asteroid locks on, then use the same controls to drive across its surface.
                                        </p>
                                    </div>
                                    <button
                                        className={`intro-close-btn${introPopupCanClose ? '' : ' locked'}`}
                                        onClick={(e) => { e.stopPropagation(); if (introPopupCanClose) closeIntroPopup(); }}
                                        disabled={!introPopupCanClose}
                                    >
                                        {introPopupCanClose ? 'Begin Mission' : 'Reading...'}
                                    </button>
                                    {introPopupCanClose && (
                                        <p className="intro-dismiss-hint">Press Enter or tap anywhere to dismiss</p>
                                    )}
                                </div>
                            </div>
                        )}

                        <div
                            id="controls"
                            style={{
                                visibility: arFlatHudChromeVisible ? 'visible' : 'hidden',
                                pointerEvents: arFlatHudChromeVisible ? 'auto' : 'none',
                            }}
                        >
                            <div
                                className="dpad-circle dpad-circle--ar"
                                onPointerDown={(e) => { e.preventDefault(); (e.target as HTMLElement).setPointerCapture(e.pointerId); updateDpadFromPointer(e); }}
                                onPointerMove={(e) => { if (e.buttons) updateDpadFromPointer(e); }}
                                onPointerUp={(e) => { (e.target as HTMLElement).releasePointerCapture(e.pointerId); clearDpadInput(); }}
                                onPointerCancel={(e) => { (e.target as HTMLElement).releasePointerCapture(e.pointerId); clearDpadInput(); }}
                            />
                        </div>
                    </div>
                </>
            )}

            {gameState === 'WEB_GAME' && (
                <>
                    {/* Web Game Scene - hidden until rover is snapped to surface */}
                    <div style={{
                        position: 'fixed', top: 0, left: 0, width: '100%', height: '100%', zIndex: 0,
                        opacity: roverReady ? 1 : 0,
                        transition: 'opacity 0.15s ease-out'
                    }}>
                        <a-scene
                            embedded
                            vr-mode-ui="enabled: false"
                            background="color: #000011"
                            renderer="colorManagement: true; logarithmicDepthBuffer: true"
                            light="defaultLightsEnabled: false"
                        >
                            <a-camera
                                id="follow-camera"
                                position="0 0 5"
                                look-controls="enabled: false"
                                wasd-controls="enabled: false"
                            ></a-camera>

                            {/* Helper markers */}
                            <a-sphere position="0 0 0" radius="0.2" color="yellow"></a-sphere>
                            <a-text value="ORIGIN" position="0 0.5 0" scale="1 1 1" color="yellow" align="center"></a-text>

                            {/* Lighting — intensities tuned for A-Frame 1.6.x physical light units (same values
                                ran much dimmer in A-Frame 1.4.2; defaultLightsEnabled:false prevents double-stacking). */}
                            <a-light type="ambient" color="#FFFFFF" intensity="0.35"></a-light>
                            <a-light type="directional" color="#FFFFFF" intensity="0.5" position="3 5 4"></a-light>
                            <a-light type="directional" color="#E8E8FF" intensity="0.25" position="-2 -3 -4"></a-light>
                            <a-light type="point" color="#FFFFFF" intensity="0.15" position="-3 2 3"></a-light>
                            <a-light type="point" color="#FFFFFF" intensity="0.15" position="0 -2 -5"></a-light>

                            {/* Space background — stars distributed across full surrounding sphere */}
                            <a-entity>
                                {STARS.map(s => (
                                    <a-sphere
                                        key={s.id}
                                        position={s.pos}
                                        radius={s.radius}
                                        color={s.color}
                                        opacity={s.opacity}
                                        material="shader: flat"
                                        animation={`property: scale; from: 1 1 1; to: 1.1 1.1 1.1; loop: true; dir: alternate; dur: ${s.dur}; delay: ${s.delay}; easing: easeInOutSine`}
                                        animation__opacity={`property: material.opacity; from: ${s.opacity}; to: ${(s.opacity * 0.3).toFixed(2)}; loop: true; dir: alternate; dur: ${Math.round(s.dur * 0.7)}; delay: ${s.delay}; easing: easeInOutSine`}
                                    ></a-sphere>
                                ))}
                            </a-entity>

                            {/* VISUAL ASTEROID (web only — AR still uses AsteroidPsyche_Collision.glb on marker) */}
                            <a-entity
                                id="asteroid"
                                position="0 0 0"
                                rotation="0 0 0"
                            >
                                <a-gltf-model
                                    id="asteroid-model"
                                    src="./models/AsteroidPsyche.glb"
                                    scale="2.5 2.5 2.5"
                                    position="-3.75 -2.2 3.22"
                                ></a-gltf-model>
                            </a-entity>

                            {/* COLLISION MESH - hidden (only used by Rust raycasting) */}
                            <a-entity
                                id="collision-viz"
                                position="0 0 0"
                                rotation="0 0 0"
                                visible="false"
                            >
                                <a-gltf-model
                                    src="./models/AsteroidPsyche_Collision.glb"
                                    scale="2.5 2.5 2.5"
                                    position="-3.75 -2.2 3.22"
                                ></a-gltf-model>
                            </a-entity>

                            {/* Samples (collectibles) */}
                            {samples.map(s => (
                                <a-entity key={s.id} position={`${s.x} ${s.y} ${s.z}`} rotation={s.rotation}>
                                    <a-gltf-model src={`./models/${s.model}.glb`} scale="0.2 0.2 0.2" />
                                </a-entity>
                            ))}

                            {/* Obstacles (visual only — collision still driven in move loop) */}
                            {obstacles.map(o => (
                                <a-entity key={o.id} position={`${o.x} ${o.y} ${o.z}`}>
                                    <a-sphere radius={o.radius} color="#ff4d4d" material="transparent: true; opacity: 0" />
                                </a-entity>
                            ))}

                            {/* Sample indicator arrow — orbits rover in tangent plane toward nearest sample */}
                            <a-entity id="sample-arrow" visible="false">
                                <a-entity animation="property: scale; from: 1 1 1; to: 1.35 1.35 1.35; loop: true; dir: alternate; dur: 500; easing: easeInOutSine">
                                    {/* Arrowhead */}
                                    <a-cone
                                        height="0.09"
                                        radius-bottom="0.05"
                                        radius-top="0"
                                        color="#FFD700"
                                        position="0 0.1 0"
                                        material="emissive: #FFD700; emissiveIntensity: 0.55; transparent: true; opacity: 0.95"
                                        animation="property: material.opacity; from: 0.95; to: 0.3; loop: true; dir: alternate; dur: 500; easing: easeInOutSine"
                                    />
                                    {/* Shaft */}
                                    <a-cylinder
                                        radius="0.013"
                                        height="0.11"
                                        color="#FFD700"
                                        position="0 0.02 0"
                                        material="emissive: #FFD700; emissiveIntensity: 0.35; transparent: true; opacity: 0.8"
                                        animation="property: material.opacity; from: 0.8; to: 0.2; loop: true; dir: alternate; dur: 500; easing: easeInOutSine"
                                    />
                                </a-entity>
                            </a-entity>

                            {/* Rover */}
                            <a-entity
                                id="rover"
                                position="0 0 3.3"
                                rotation="0 0 0"
                                scale="1.2 1.2 1.2"
                                visible={roverReady ? "true" : "false"}
                            >
                                <a-gltf-model
                                    src="./models/craft_racer.glb"
                                    scale="0.2 0.2 0.2"
                                ></a-gltf-model>

                                {/* Original rover — saved as primitive A-Frame shapes
                                <a-box width="0.1" height="0.16" depth="0.52" color="#2A2A2A" position="-0.25 -0.04 0"></a-box>
                                <a-box width="0.1" height="0.16" depth="0.52" color="#2A2A2A" position="0.25 -0.04 0"></a-box>
                                <a-cylinder radius="0.08" height="0.1" rotation="0 0 90" color="#3A3A3A" position="-0.25 -0.04 -0.2"></a-cylinder>
                                <a-cylinder radius="0.08" height="0.1" rotation="0 0 90" color="#3A3A3A" position="-0.25 -0.04 0.2"></a-cylinder>
                                <a-cylinder radius="0.08" height="0.1" rotation="0 0 90" color="#3A3A3A" position="0.25 -0.04 -0.2"></a-cylinder>
                                <a-cylinder radius="0.08" height="0.1" rotation="0 0 90" color="#3A3A3A" position="0.25 -0.04 0.2"></a-cylinder>
                                <a-box width="0.4" height="0.32" depth="0.36" color="#B8963E" position="0 0.14 0"></a-box>
                                <a-box width="0.38" height="0.28" depth="0.01" color="#8B7230" position="0 0.15 -0.18"></a-box>
                                <a-box width="0.38" height="0.28" depth="0.01" color="#8B7230" position="0 0.15 0.18"></a-box>
                                <a-box width="0.42" height="0.02" depth="0.38" color="#9E8438" position="0 0.31 0"></a-box>
                                <a-cylinder radius="0.025" height="0.18" color="#707070" position="0 0.41 -0.04"></a-cylinder>
                                <a-cylinder radius="0.025" height="0.18" color="#707070" position="0 0.41 -0.04" rotation="0 0 6"></a-cylinder>
                                <a-box width="0.26" height="0.07" depth="0.07" color="#606060" position="0 0.52 -0.06"></a-box>
                                <a-cylinder radius="0.055" height="0.14" rotation="90 0 0" color="#505050" position="-0.08 0.52 -0.14"></a-cylinder>
                                <a-cylinder radius="0.055" height="0.14" rotation="90 0 0" color="#505050" position="0.08 0.52 -0.14"></a-cylinder>
                                <a-cylinder radius="0.058" height="0.02" rotation="90 0 0" color="#404040" position="-0.08 0.52 -0.21"></a-cylinder>
                                <a-cylinder radius="0.058" height="0.02" rotation="90 0 0" color="#404040" position="0.08 0.52 -0.21"></a-cylinder>
                                <a-sphere radius="0.048" color="#6DB8D4" position="-0.08 0.52 -0.22"></a-sphere>
                                <a-sphere radius="0.048" color="#6DB8D4" position="0.08 0.52 -0.22"></a-sphere>
                                <a-sphere radius="0.025" color="#1A1A1A" position="-0.08 0.52 -0.25"></a-sphere>
                                <a-sphere radius="0.025" color="#1A1A1A" position="0.08 0.52 -0.25"></a-sphere>
                                <a-box width="0.035" height="0.035" depth="0.18" color="#707070" rotation="15 0 0" position="-0.24 0.14 -0.14"></a-box>
                                <a-box width="0.035" height="0.035" depth="0.18" color="#707070" rotation="15 0 0" position="0.24 0.14 -0.14"></a-box>
                                <a-box width="0.06" height="0.02" depth="0.06" color="#606060" rotation="15 0 0" position="-0.24 0.14 -0.25"></a-box>
                                <a-box width="0.06" height="0.02" depth="0.06" color="#606060" rotation="15 0 0" position="0.24 0.14 -0.25"></a-box>
                                <a-box width="0.08" height="0.02" depth="0.2" color="#555555" position="0 0.33 0"></a-box>
                                */}
                            </a-entity>
                        </a-scene>
                    </div>

                    <div id="ui-overlay" style={{ display: 'block' }}>
                        <div className="hud-stack">
                            {modeCfg.energyEnabled && (
                                <div className="energy-display">
                                    ENERGY
                                    <div className="energy-bar"><div style={{ width: `${(energy / MAX_ENERGY) * 100}%` }} /></div>
                                </div>
                            )}
                            <div id="score-display">
                                SCORE <span id="score">{score}</span>
                            </div>
                            <div className="samples-display">
                                SAMPLES <span className="samples-value">{samplesCollected}</span>
                            </div>
                        </div>
                        {/* SAMPLE POPUP */}
                        <div
                            className={`sample-overlay ${waypointPopup ? 'open' : 'closed'}`}
                            onClick={() => setWaypointPopup(null)}
                        >
                            <div
                                className="sample-modal"
                                onClick={(e) => e.stopPropagation()}
                                role="dialog"
                                aria-modal="true"
                                aria-hidden={!waypointPopup}
                            >
                                {waypointPopup?.image && (
                                    <div className="sample-image-panel">
                                        <img src={waypointPopup.image} alt="Waypoint visual" />
                                    </div>
                                )}
                                <div className="sample-text-panel">
                                    <h2 className="sample-title">{waypointPopup?.title}</h2>
                                    {waypointPopup?.body && (
                                        <p className="sample-body">{waypointPopup.body}</p>
                                    )}
                                    <button
                                        ref={sampleContinueBtnRef}
                                        className="sample-continue-btn"
                                        onClick={() => setWaypointPopup(null)}
                                    >
                                        Continue
                                    </button>
                                    <div className="sample-hint">Press Space or click outside to continue</div>
                                </div>
                            </div>
                        </div>
                        {/* END SCREEN */}
                        {showEndScreen && (
                            <div className="end-overlay" role="dialog" aria-modal="true">
                                <div className="end-modal">
                                    <h2 className="end-title">
                                        {endReason === 'complete' ? 'Mission Complete!' : 'Out of Energy'}
                                    </h2>
                                    <p className="end-subtitle">
                                        {endReason === 'complete'
                                            ? 'All samples have been recovered from the surface of Psyche.'
                                            : "Your rover's battery has been depleted. Mission over."}
                                    </p>

                                    <div className="end-stats">
                                        <div className="end-stat">
                                            <span className="end-stat-label">Samples Collected</span>
                                            <span className="end-stat-value">{samplesCollected} / {modeCfg.spawnSamples}</span>
                                        </div>
                                        {energyBonus > 0 && (
                                            <div className="end-stat">
                                                <span className="end-stat-label">Energy Bonus</span>
                                                <span className="end-stat-value">+{energyBonus}</span>
                                            </div>
                                        )}
                                        <div className="end-stat">
                                            <span className="end-stat-label">Final Score</span>
                                            <span className="end-stat-value">{score}</span>
                                        </div>
                                    </div>

                                    <button className="end-menu-btn" onClick={returnToMenu}>
                                        Return to Main Menu
                                    </button>
                                </div>
                            </div>
                        )}
                        {/* INTRO POPUP */}
                        {showIntroPopup && (
                            <div
                                className="intro-overlay"
                                onClick={() => { if (introPopupCanClose) closeIntroPopup(); }}
                                role="dialog"
                                aria-modal="true"
                            >
                                <div className="intro-modal">
                                    <h2 className="intro-title">{INTRO_CONTENT[difficulty].welcome}</h2>

                                    <div className="intro-section">
                                        <h3 className="intro-section-heading">Controls</h3>
                                        <div className="intro-controls-grid">
                                            <span className="key-hint">W / A / S / D</span><span>Move rover</span>
                                            <span className="key-hint">Arrow Keys</span><span>Move rover</span>
                                            <span className="key-hint">D-pad</span><span>Move rover (mobile/touch)</span>
                                        </div>
                                    </div>

                                    <div className="intro-section">
                                        <p className="intro-description">{INTRO_CONTENT[difficulty].description}</p>
                                    </div>

                                    <button
                                        className={`intro-close-btn${introPopupCanClose ? '' : ' locked'}`}
                                        onClick={(e) => { e.stopPropagation(); if (introPopupCanClose) closeIntroPopup(); }}
                                        disabled={!introPopupCanClose}
                                    >
                                        {introPopupCanClose ? 'Begin Mission' : 'Reading...'}
                                    </button>
                                    {introPopupCanClose && (
                                        <p className="intro-dismiss-hint">Press Space or click outside to continue</p>
                                    )}
                                </div>
                            </div>
                        )}
                        <div id="controls">
                            <div
                                className="dpad-circle"
                                onPointerDown={(e) => { e.preventDefault(); (e.target as HTMLElement).setPointerCapture(e.pointerId); updateDpadFromPointer(e); }}
                                onPointerMove={(e) => { if (e.buttons) updateDpadFromPointer(e); }}
                                onPointerUp={(e) => { (e.target as HTMLElement).releasePointerCapture(e.pointerId); clearDpadInput(); }}
                                onPointerCancel={(e) => { (e.target as HTMLElement).releasePointerCapture(e.pointerId); clearDpadInput(); }}
                            />
                        </div>
                    </div>
                </>
            )}
        </div>
    );
};

export default App;
