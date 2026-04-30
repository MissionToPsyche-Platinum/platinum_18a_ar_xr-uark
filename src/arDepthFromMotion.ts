/**
 * In-house depth-from-motion fusion (concept from ARCore Depth / depth-from-motion):
 * https://developers.google.com/ar/develop/depth
 *
 * Browser + AR.js cannot access the native ARCore Depth API. We approximate it by sampling
 * several consecutive camera frames while the phone micro-moves, then combining temporal
 * variance with first-vs-last frame difference as a parallax proxy. The result is fused into
 * the luminance heightmap so rover physics and the displaced mesh better track relief, and we
 * emit a jet colormap PNG for a small “see depth” HUD.
 */

export type CropRect = { x: number; y: number; size: number };

export type HeightmapMotionTarget = {
    data: Float32Array;
    size: number;
    cropRect: CropRect;
    /** Filled after fusion — data URL of jet heatmap (motion-depth proxy). */
    depthDebugUrl?: string;
};

function sleep(ms: number): Promise<void> {
    return new Promise((r) => setTimeout(r, ms));
}

function inDisk(x: number, y: number, outSize: number): boolean {
    const dx = x - outSize / 2 + 0.5;
    const dy = y - outSize / 2 + 0.5;
    return dx * dx + dy * dy <= (outSize / 2 - 0.5) ** 2;
}

/** Match luminance heightmap normalization: disk-only mean/std → squash around 0.5. */
function renormalizeHeightDisk(data: Float32Array, outSize: number): void {
    let sum = 0;
    let sum2 = 0;
    let count = 0;
    for (let y = 0; y < outSize; y++) {
        for (let x = 0; x < outSize; x++) {
            if (!inDisk(x, y, outSize)) continue;
            const v = data[y * outSize + x];
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
            const i = y * outSize + x;
            const z = (data[i] - mean) / std;
            data[i] = Math.max(0, Math.min(1, 0.5 + z * 0.25));
        }
    }
}

/** Classic jet colormap, t ∈ [0,1]. */
function jetRgb(t: number): [number, number, number] {
    t = Math.max(0, Math.min(1, t));
    const four = t * 4;
    const seg = Math.min(3, Math.floor(four));
    const f = four - seg;
    switch (seg) {
        case 0:
            return [0, Math.round(255 * f), 255];
        case 1:
            return [Math.round(255 * f), 255, Math.round(255 * (1 - f))];
        case 2:
            return [255, Math.round(255 * (1 - f)), 0];
        default:
            return [Math.round(255 * (1 - f)), 0, 0];
    }
}

function motionNormToDataUrl(motionNorm: Float32Array, outSize: number): string {
    const c = document.createElement('canvas');
    c.width = outSize;
    c.height = outSize;
    const ctx = c.getContext('2d');
    if (!ctx) return '';
    const img = ctx.createImageData(outSize, outSize);
    for (let y = 0; y < outSize; y++) {
        for (let x = 0; x < outSize; x++) {
            const i = (y * outSize + x) * 4;
            const t = motionNorm[y * outSize + x];
            if (!inDisk(x, y, outSize)) {
                img.data[i] = 0;
                img.data[i + 1] = 0;
                img.data[i + 2] = 0;
                img.data[i + 3] = 0;
                continue;
            }
            const [r, g, b] = jetRgb(t);
            img.data[i] = r;
            img.data[i + 1] = g;
            img.data[i + 2] = b;
            img.data[i + 3] = 235;
        }
    }
    ctx.putImageData(img, 0, 0);
    return c.toDataURL('image/png');
}

/**
 * Samples `frameCount` grayscale crops (same rect as the heightmap), spaced by `spacingMs`,
 * fuses a motion-depth proxy into `target.data`, re-normalizes, and sets `target.depthDebugUrl`.
 */
export async function augmentHeightmapWithDepthFromMotion(
    cv: any,
    video: HTMLVideoElement,
    target: HeightmapMotionTarget,
    opts?: { frameCount?: number; spacingMs?: number; motionBlend?: number },
): Promise<void> {
    const frameCount = Math.max(2, Math.min(8, opts?.frameCount ?? 4));
    const spacingMs = Math.max(24, opts?.spacingMs ?? 72);
    const motionBlend = Math.max(0, Math.min(0.85, opts?.motionBlend ?? 0.38));

    const { cropRect, data, size } = target;
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!cv || !vw || !vh || cropRect.size < 16 || size < 8) return;

    const canvas = document.createElement('canvas');
    canvas.width = vw;
    canvas.height = vh;
    const ctx2d = canvas.getContext('2d');
    if (!ctx2d) return;

    const mats: any[] = [];
    try {
        for (let f = 0; f < frameCount; f++) {
            if (f > 0) await sleep(spacingMs);
            ctx2d.drawImage(video, 0, 0, vw, vh);
            const full = cv.imread(canvas);
            const roi = full.roi(new cv.Rect(cropRect.x, cropRect.y, cropRect.size, cropRect.size));
            const gray = new cv.Mat();
            cv.cvtColor(roi, gray, cv.COLOR_RGBA2GRAY);
            const small = new cv.Mat();
            cv.resize(gray, small, new cv.Size(size, size), 0, 0, cv.INTER_AREA);
            mats.push(small);
            roi.delete();
            gray.delete();
            full.delete();
        }

        const n = size * size;
        const motion = new Float32Array(n);
        const u0 = mats[0].data as Uint8Array;
        const uLast = mats[mats.length - 1].data as Uint8Array;

        for (let i = 0; i < n; i++) {
            let sum = 0;
            let sum2 = 0;
            for (const m of mats) {
                const v = (m.data as Uint8Array)[i] / 255;
                sum += v;
                sum2 += v * v;
            }
            const mean = sum / mats.length;
            const varv = Math.max(0, sum2 / mats.length - mean * mean);
            const stdT = Math.sqrt(varv);
            const parallax = Math.abs(u0[i] - uLast[i]) / 255;
            motion[i] = stdT * 0.55 + parallax * 0.45;
        }

        let mn = Infinity;
        let mx = -Infinity;
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                if (!inDisk(x, y, size)) continue;
                const v = motion[y * size + x];
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
        }
        const span = mx - mn || 1;
        const motionNorm = new Float32Array(n);
        for (let i = 0; i < n; i++) {
            motionNorm[i] = (motion[i] - mn) / span;
        }

        for (let i = 0; i < n; i++) {
            data[i] = Math.max(0, Math.min(1, data[i] * (1 - motionBlend) + motionNorm[i] * motionBlend));
        }
        renormalizeHeightDisk(data, size);

        target.depthDebugUrl = motionNormToDataUrl(motionNorm, size);
    } catch (e) {
        console.warn('[AR][depth-from-motion] fusion skipped:', e);
    } finally {
        for (const m of mats) {
            try {
                m.delete();
            } catch {
                /* ignore */
            }
        }
    }
}
