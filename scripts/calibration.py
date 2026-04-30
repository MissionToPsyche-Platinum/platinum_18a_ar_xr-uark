import argparse
import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import cv2
import numpy as np

# Project marker layout
ALL_IDS = set(range(8))
ORIGIN_ID = 4
TABLE_IDS = {1, 3, 4, 6}
SURFACE_IDS = {0, 2, 5, 7}
REF_BASE_URL = (
    "https://raw.githubusercontent.com/nicolocarpignoli/"
    "artoolkit-barcode-markers-collection/master/3x3_hamming_6_3"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Desktop calibration for NASA Psyche AR config.json")
    parser.add_argument("--camera", type=int, default=1, help="Camera index (default: 1)")
    parser.add_argument(
        "--marker-length",
        type=float,
        default=0.0508,
        help="Printed marker black-square width in meters (default: 0.0508 for 2-inch prints)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "public" / "config.json"),
        help="Output config.json path",
    )
    parser.add_argument(
        "--refs-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / ".marker_refs_3x3_hamming63"),
        help="Directory used to cache ARToolKit marker reference PNG files",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.28,
        help="Max binary mismatch ratio for marker match acceptance (default: 0.20)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="Minimum accepted samples per marker for export (except origin)",
    )
    parser.add_argument(
        "--sample-score-max",
        type=float,
        default=0.28,
        help="Per-frame sample acceptance threshold (lower is stricter)",
    )
    parser.add_argument(
        "--max-median-score",
        type=float,
        default=0.20,
        help="Export marker only if median sample score <= this value",
    )
    parser.add_argument(
        "--target-ids",
        type=str,
        default="all",
        help='Marker IDs to update, e.g. "0,2,5,7" or "all"',
    )
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="Merge updated IDs into existing output config instead of overwriting all IDs",
    )
    parser.add_argument(
        "--min-surface-y",
        type=float,
        default=0.02,
        help="Minimum exported Y for known surface markers in meters",
    )
    return parser.parse_args()


def mat4_from_rt(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rmat, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3, 1)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rmat
    out[:3, 3:4] = t
    return out


def orthonormalize_rotation(r: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(r)
    out = u @ vt
    if np.linalg.det(out) < 0:
        u[:, -1] *= -1
        out = u @ vt
    return out


def flatten_for_three(m: np.ndarray) -> list[float]:
    # THREE.Matrix4 elements are column-major.
    return m.T.reshape(-1).astype(float).tolist()


def apply_basis_opencv_to_three(m_cv: np.ndarray) -> np.ndarray:
    # OpenCV camera frame: +X right, +Y down, +Z forward
    # Three.js frame:       +X right, +Y up,   +Z toward viewer
    b = np.eye(4, dtype=np.float64)
    b[1, 1] = -1.0
    b[2, 2] = -1.0
    return b @ m_cv @ b


def align_table_rotation_to_plane(m: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    """
    Force marker orientation to lie flat on measured table plane,
    preserving each marker's in-plane heading as much as possible.
    """
    out = m.copy()
    up = plane_normal.astype(np.float64)
    up /= np.linalg.norm(up) + 1e-12

    # Marker local X axis projected onto measured table plane.
    x = out[:3, 0].astype(np.float64)
    x = x - np.dot(x, up) * up
    x_len = float(np.linalg.norm(x))
    if x_len < 1e-8:
        # Fallback to projected Z axis and rotate 90deg to reconstruct X.
        z0 = out[:3, 2].astype(np.float64)
        z0 = z0 - np.dot(z0, up) * up
        z_len = float(np.linalg.norm(z0))
        if z_len < 1e-8:
            x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            z0 /= z_len
            x = np.cross(up, z0)
            x /= np.linalg.norm(x) + 1e-12
    else:
        x /= x_len

    # Right-handed orthonormal basis with strict table normal as "up".
    z = np.cross(x, up)
    z /= np.linalg.norm(z) + 1e-12
    x = np.cross(up, z)
    x /= np.linalg.norm(x) + 1e-12

    out[:3, 0] = x
    out[:3, 1] = up
    out[:3, 2] = z
    return out


def mean_pose(mats: list[np.ndarray]) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    if not mats:
        return out
    ts = np.array([m[:3, 3] for m in mats], dtype=np.float64)
    out[:3, 3] = np.median(ts, axis=0)
    r_sum = np.zeros((3, 3), dtype=np.float64)
    for m in mats:
        r_sum += m[:3, :3]
    out[:3, :3] = orthonormalize_rotation(r_sum)
    return out


def fit_plane(points: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    pts = np.array(points, dtype=np.float64)
    c = np.mean(pts, axis=0)
    _, _, vh = np.linalg.svd(pts - c)
    n = vh[-1]
    n = n / (np.linalg.norm(n) + 1e-12)
    # Prefer +Y hemisphere for stable visualization/reporting.
    if n[1] < 0:
        n = -n
    return n, c


def rotation_align_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return 3x3 rotation matrix mapping unit vector a to unit vector b."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a /= np.linalg.norm(a) + 1e-12
    b /= np.linalg.norm(b) + 1e-12
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = float(np.dot(a, b))
    if s < 1e-9:
        if c > 0:
            return np.eye(3, dtype=np.float64)
        # 180-degree flip: choose any axis orthogonal to a.
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(axis, a)) > 0.9:
            axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        axis = axis - np.dot(axis, a) * a
        axis /= np.linalg.norm(axis) + 1e-12
        # Rodrigues with theta=pi: R = -I + 2 uu^T
        return -np.eye(3, dtype=np.float64) + 2.0 * np.outer(axis, axis)
    vx = np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )
    r = np.eye(3, dtype=np.float64) + vx + (vx @ vx) * ((1.0 - c) / (s * s))
    return orthonormalize_rotation(r)


def project_point_to_plane(point: np.ndarray, normal: np.ndarray, plane_point: np.ndarray) -> np.ndarray:
    v = point - plane_point
    return point - np.dot(v, normal) * normal


def parse_target_ids(raw: str) -> set[int] | None:
    raw = (raw or "").strip().lower()
    if raw in ("", "all", "*"):
        return None
    out: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        out.add(int(token))
    return out


def controls_to_map(controls: list[dict]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for row in controls:
        bid = row.get("parameters", {}).get("barcodeValue")
        if isinstance(bid, int):
            out[bid] = row
    return out


def merge_with_existing_config(output_path: Path, new_controls: list[dict], target_ids: set[int] | None) -> list[dict]:
    base_controls: list[dict] = []
    if output_path.exists():
        try:
            old_json = json.loads(output_path.read_text(encoding="utf-8"))
            base_controls = old_json.get("subMarkersControls", []) if isinstance(old_json, dict) else []
        except Exception:
            base_controls = []

    base_map = controls_to_map(base_controls)
    new_map = controls_to_map(new_controls)

    if target_ids is None:
        # Replace all IDs present in new export.
        for marker_id, row in new_map.items():
            base_map[marker_id] = row
    else:
        for marker_id in target_ids:
            row = new_map.get(marker_id)
            if row is not None:
                base_map[marker_id] = row

    # Keep output deterministically sorted by marker id.
    return [base_map[k] for k in sorted(base_map.keys())]


def ensure_reference_image(path: Path, marker_id: int) -> np.ndarray:
    if not path.exists():
        url = f"{REF_BASE_URL}/{marker_id}.png"
        try:
            with urlopen(url, timeout=12) as resp:
                data = resp.read()
            path.write_bytes(data)
        except URLError as exc:
            raise RuntimeError(
                f"Could not download ARToolKit marker reference {marker_id} from {url}. "
                f"Check internet access or pre-place {path}."
            ) from exc

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read reference marker image: {path}")
    return img


def preprocess_marker_image(gray: np.ndarray, size: int) -> np.ndarray:
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    _, bw = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw


def load_reference_markers(refs_dir: Path, size: int) -> dict[int, np.ndarray]:
    refs_dir.mkdir(parents=True, exist_ok=True)
    refs: dict[int, np.ndarray] = {}
    for marker_id in sorted(ALL_IDS):
        p = refs_dir / f"{marker_id}.png"
        img = ensure_reference_image(p, marker_id)
        refs[marker_id] = preprocess_marker_image(img, size)
    return refs


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    # Returns corners in image order: TL, TR, BR, BL.
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def dedupe_quads(quads: list[np.ndarray]) -> list[np.ndarray]:
    if not quads:
        return quads
    out: list[np.ndarray] = []
    seen: list[tuple[float, float, float]] = []
    for q in quads:
        c = np.mean(q, axis=0)
        area = abs(cv2.contourArea(q.astype(np.float32)))
        key = (float(c[0]), float(c[1]), float(area))
        keep = True
        for sx, sy, sa in seen:
            if (sx - key[0]) ** 2 + (sy - key[1]) ** 2 < 16 * 16 and abs(sa - key[2]) / max(sa, 1e-6) < 0.2:
                keep = False
                break
        if keep:
            seen.append(key)
            out.append(q)
    return out


def quad_candidates(gray: np.ndarray) -> list[np.ndarray]:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thrs = [
        cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7),
        cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 41, 9),
    ]
    all_contours = []
    for thr in thrs:
        k = np.ones((3, 3), np.uint8)
        thr2 = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, k, iterations=1)
        contours, _ = cv2.findContours(thr2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        all_contours.extend(contours)

    h, w = gray.shape[:2]
    min_area = (h * w) * 0.00025
    max_area = (h * w) * 0.8
    out: list[np.ndarray] = []
    for c in all_contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        q = order_quad_points(approx)
        side_a = np.linalg.norm(q[1] - q[0]) + 1e-6
        side_b = np.linalg.norm(q[2] - q[1]) + 1e-6
        ratio = max(side_a, side_b) / min(side_a, side_b)
        if ratio > 1.35:
            continue
        out.append(q)
    return dedupe_quads(out)


def warp_quad(gray: np.ndarray, quad: np.ndarray, size: int) -> np.ndarray:
    dst = np.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
        dtype=np.float32,
    )
    m = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(gray, m, (size, size))


def border_is_black_enough(bw: np.ndarray) -> bool:
    size = bw.shape[0]
    b = max(4, int(size * 0.10))
    edge = np.concatenate([bw[:b, :], bw[-b:, :], bw[:, :b], bw[:, -b:]], axis=None)
    black_ratio = float(np.mean(edge < 128))
    return black_ratio > 0.65


def extract_3x3_bits(bw: np.ndarray, border_ratio: float) -> tuple[int, ...]:
    n = bw.shape[0]
    start = int(round(border_ratio * n))
    end = int(round((1.0 - border_ratio) * n))
    span = max(end - start, 3)
    step = span / 3.0
    bits = []
    for r in range(3):
        for c in range(3):
            cy = start + (r + 0.5) * step
            cx = start + (c + 0.5) * step
            half = max(1, int(step * 0.22))
            y0 = max(0, int(round(cy)) - half)
            y1 = min(n, int(round(cy)) + half + 1)
            x0 = max(0, int(round(cx)) - half)
            x1 = min(n, int(round(cx)) + half + 1)
            patch = bw[y0:y1, x0:x1]
            bits.append(1 if float(np.mean(patch)) < 128 else 0)
    return tuple(bits)


def hamming(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


def build_reference_bits(refs: dict[int, np.ndarray]) -> dict[int, tuple[int, ...]]:
    # 0.24 is the intended ARToolKit border for this project.
    return {marker_id: extract_3x3_bits(ref_bw, border_ratio=0.24) for marker_id, ref_bw in refs.items()}


def match_marker(
    cand_bw: np.ndarray,
    refs: dict[int, np.ndarray],
    ref_bits: dict[int, tuple[int, ...]],
    mismatch_threshold: float,
) -> tuple[int | None, int, float]:
    # Tolerate marker print/generator border variants.
    cand_bits_by_ratio = [
        np.array(extract_3x3_bits(cand_bw, border_ratio=r)).reshape(3, 3)
        for r in (0.24, 0.25, 0.20)
    ]
    best_id = None
    best_rot = 0
    best_score = 1.0
    best_ham = 999
    for marker_id, ref_bw in refs.items():
        rb = ref_bits[marker_id]
        for rot in range(4):
            # rot is CCW rotations to align candidate with reference.
            c = np.rot90(cand_bw, rot)
            score = float(np.mean(c != ref_bw))
            ham = 999
            for cand_bits_33 in cand_bits_by_ratio:
                cb = np.rot90(cand_bits_33, rot).reshape(-1)
                ham = min(ham, hamming(tuple(int(v) for v in cb.tolist()), rb))
            if ham < best_ham or (ham == best_ham and score < best_score):
                best_id = marker_id
                best_rot = rot
                best_score = score
                best_ham = ham

    # Hamming-robust gate first, pixel mismatch as secondary confidence.
    if best_id is None or best_ham > 2 or best_score > mismatch_threshold:
        return None, 0, best_score
    return best_id, best_rot, best_score


def rotate_corner_order_to_canonical(corners_tl_tr_br_bl: np.ndarray, rot_ccw: int) -> np.ndarray:
    # If candidate needs rot_ccw CCW to match reference, shift corner order right by rot_ccw.
    # This maps image corners back to marker canonical TL,TR,BR,BL order for solvePnP.
    return np.roll(corners_tl_tr_br_bl, shift=rot_ccw, axis=0).astype(np.float32)


def detect_marker_poses(frame, refs, ref_bits, camera_matrix, dist_coeffs, marker_points, mismatch_threshold):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    candidates = quad_candidates(gray)

    detections = []
    by_id = {}
    warp_size = next(iter(refs.values())).shape[0]
    for quad in candidates:
        patch = warp_quad(gray, quad, warp_size)
        bw = preprocess_marker_image(patch, warp_size)
        if not border_is_black_enough(bw):
            continue

        marker_id, rot_ccw, score = match_marker(bw, refs, ref_bits, mismatch_threshold)
        if marker_id is None:
            continue

        corners = rotate_corner_order_to_canonical(quad, rot_ccw)
        ok, rvec, tvec = cv2.solvePnP(
            marker_points,
            corners,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok:
            continue

        det = {"id": int(marker_id), "score": float(score), "corners": corners}
        detections.append(det)

        prev = by_id.get(marker_id)
        if prev is None or score < prev["score"]:
            by_id[marker_id] = {"score": score, "rvec": rvec, "tvec": tvec, "corners": corners}

    poses = {int(k): (v["rvec"], v["tvec"]) for k, v in by_id.items()}
    pose_scores = {int(k): float(v["score"]) for k, v in by_id.items()}
    return poses, detections, pose_scores


def camera_marker_mats_from_poses(poses: dict[int, tuple[np.ndarray, np.ndarray]]) -> dict[int, np.ndarray]:
    """Build T_cam_marker for currently visible markers."""
    out: dict[int, np.ndarray] = {}
    for marker_id, (rvec, tvec) in poses.items():
        out[int(marker_id)] = mat4_from_rt(rvec, tvec)
    return out


def estimate_root_mats_from_visible(
    visible_cam_mats: dict[int, np.ndarray],
    known_root_mats: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    """
    Estimate root->marker transforms for visible markers using any visible known anchor.
    This enables chain expansion (e.g. 4->1 then 1->0) instead of requiring 4 every frame.
    """
    if not visible_cam_mats:
        return {}
    visible_known = [mid for mid in visible_cam_mats.keys() if mid in known_root_mats]
    if not visible_known:
        return {}

    estimates: dict[int, list[np.ndarray]] = {}
    for anchor_id in visible_known:
        t_root_anchor = known_root_mats[anchor_id]
        t_cam_anchor = visible_cam_mats[anchor_id]
        inv_t_cam_anchor = np.linalg.inv(t_cam_anchor)
        for marker_id, t_cam_marker in visible_cam_mats.items():
            t_anchor_marker = inv_t_cam_anchor @ t_cam_marker
            t_root_marker = t_root_anchor @ t_anchor_marker
            estimates.setdefault(marker_id, []).append(t_root_marker)

    fused: dict[int, np.ndarray] = {}
    for marker_id, mats in estimates.items():
        fused[marker_id] = mean_pose(mats)
    return fused


def build_config_from_samples(
    samples: dict[int, list[np.ndarray]],
    sample_scores: dict[int, list[float]],
    min_samples: int,
    max_median_score: float,
) -> tuple[dict, str]:
    if ORIGIN_ID not in samples or len(samples[ORIGIN_ID]) == 0:
        raise RuntimeError(f"Origin marker {ORIGIN_ID} has no collected samples.")

    averaged: dict[int, np.ndarray] = {}
    rejected: list[str] = []
    for marker_id, mats in samples.items():
        if not mats:
            continue
        if marker_id == ORIGIN_ID:
            averaged[int(marker_id)] = mean_pose(mats)
            continue
        if len(mats) < min_samples:
            rejected.append(f"{marker_id}:few({len(mats)})")
            continue
        scores = sample_scores.get(marker_id, [])
        med = float(np.median(scores)) if scores else 1.0
        if med > max_median_score:
            rejected.append(f"{marker_id}:score({med:.3f})")
            continue
        averaged[int(marker_id)] = mean_pose(mats)

    # Marker 4 defines frame origin exactly.
    averaged[ORIGIN_ID] = np.eye(4, dtype=np.float64)

    table_present = [mid for mid in sorted(TABLE_IDS) if mid in averaged]
    plane_msg = "table plane not fitted"
    if len(table_present) >= 3:
        pts = []
        for mid in table_present:
            pts.extend([m[:3, 3] for m in samples.get(mid, [])])
        if len(pts) >= 3:
            n, c = fit_plane(pts)
            r_align = rotation_align_a_to_b(n, np.array([0.0, 1.0, 0.0], dtype=np.float64))

            # Rotate whole solved layout so fitted table normal becomes +Y.
            for mid in list(averaged.keys()):
                m = averaged[mid].copy()
                m[:3, 3] = r_align @ m[:3, 3]
                m[:3, :3] = orthonormalize_rotation(r_align @ m[:3, :3])
                averaged[mid] = m

            # Table plane point also rotates; then we enforce final Y=0 consistency.
            c_aligned = r_align @ c
            up_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            plane_msg = f"table normal≈({n[0]:.3f},{n[1]:.3f},{n[2]:.3f}) -> aligned to +Y using {len(pts)} pts"
            for mid in table_present:
                m = averaged[mid].copy()
                m[:3, 3] = project_point_to_plane(m[:3, 3], up_y, c_aligned)
                m[:3, 3][1] = 0.0
                m = align_table_rotation_to_plane(m, up_y)
                m[:3, :3] = orthonormalize_rotation(m[:3, :3])
                averaged[mid] = m

    # Keep table markers on ground plane.
    for mid in TABLE_IDS:
        if mid not in averaged:
            continue
        m = averaged[mid].copy()
        m[1, 3] = 0.0
        averaged[mid] = m

    # Keep origin strict identity after all adjustments.
    averaged[ORIGIN_ID] = np.eye(4, dtype=np.float64)

    controls = []
    for marker_id in sorted(ALL_IDS):
        m = averaged.get(marker_id)
        if m is None:
            continue
        controls.append(
            {
                "parameters": {"type": "barcode", "barcodeValue": int(marker_id)},
                "poseMatrix": flatten_for_three(m),
            }
        )
    quality_msg = f"exported={sorted(int(c['parameters']['barcodeValue']) for c in controls)}"
    if rejected:
        quality_msg += f"; rejected={rejected}"
    return {"subMarkersControls": controls}, f"{plane_msg} | {quality_msg}"


def save_config(
    output_path: Path,
    samples: dict[int, list[np.ndarray]],
    sample_scores: dict[int, list[float]],
    min_samples: int,
    max_median_score: float,
    target_ids: set[int] | None,
    merge_existing: bool,
    min_surface_y: float,
) -> None:
    cfg, msg = build_config_from_samples(samples, sample_scores, min_samples, max_median_score)
    if len(cfg["subMarkersControls"]) < 2:
        raise RuntimeError("Need at least 2 reliable markers before saving.")
    new_controls = cfg["subMarkersControls"]

    # Keep known surface markers above table (Y > 0), since table markers are defined on Y=0.
    for row in new_controls:
        mid = row.get("parameters", {}).get("barcodeValue")
        if mid in SURFACE_IDS and isinstance(row.get("poseMatrix"), list) and len(row["poseMatrix"]) >= 16:
            y = float(row["poseMatrix"][13])
            row["poseMatrix"][13] = max(abs(y), float(min_surface_y))

    if target_ids is not None:
        new_controls = [
            row
            for row in new_controls
            if row.get("parameters", {}).get("barcodeValue") in target_ids
        ]
        if not new_controls:
            raise RuntimeError(f"No solved markers for requested target IDs: {sorted(target_ids)}")

    if merge_existing:
        merged = merge_with_existing_config(output_path, new_controls, target_ids)
        out_cfg = {"subMarkersControls": merged}
        output_path.write_text(json.dumps(out_cfg, indent=2), encoding="utf-8")
        print(
            f"Saved {output_path} | {msg} | merged IDs="
            f"{sorted([r['parameters']['barcodeValue'] for r in new_controls])}"
        )
    else:
        out_cfg = {"subMarkersControls": new_controls}
        output_path.write_text(json.dumps(out_cfg, indent=2), encoding="utf-8")
        print(
            f"Saved {output_path} | {msg} | wrote IDs="
            f"{sorted([r['parameters']['barcodeValue'] for r in new_controls])}"
        )


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    refs_dir = Path(args.refs_dir)
    target_ids = parse_target_ids(args.target_ids)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Approximate camera intrinsics for 1080p (replace with calibrated values if available).
    fx, fy, cx, cy = 1000.0, 1000.0, 960.0, 540.0
    camera_matrix = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)

    refs = load_reference_markers(refs_dir=refs_dir, size=160)
    ref_bits = build_reference_bits(refs)

    marker_points = np.array(
        [
            [-args.marker_length / 2, args.marker_length / 2, 0],
            [args.marker_length / 2, args.marker_length / 2, 0],
            [args.marker_length / 2, -args.marker_length / 2, 0],
            [-args.marker_length / 2, -args.marker_length / 2, 0],
        ],
        dtype=np.float32,
    )

    print("Desktop calibration started.")
    print("Controls: [s] save config.json from samples, [c] clear samples, [q] save+quit")
    print(f"Camera index: {args.camera}")
    print(f"Output path: {output_path}")
    print(f"Reference cache: {refs_dir}")
    print("Detector: ARToolKit matrix 3x3_HAMMING63, border 0.24 (template matching)")
    print(f"Origin marker: {ORIGIN_ID}; table markers constrained by fitted plane: {sorted(TABLE_IDS)}")
    print(f"Target IDs: {sorted(target_ids) if target_ids is not None else 'all'}")
    print(f"Merge mode: {args.merge_existing}")
    print(
        f"Quality gates: min_samples={args.min_samples}, "
        f"sample_score_max={args.sample_score_max}, max_median_score={args.max_median_score}"
    )

    # Multi-frame sample buffer for robust calibration.
    samples: dict[int, list[np.ndarray]] = {i: [] for i in sorted(ALL_IDS)}
    sample_scores: dict[int, list[float]] = {i: [] for i in sorted(ALL_IDS)}
    max_samples_per_marker = 240
    # Running map root(marker4)->marker in OpenCV coordinates.
    known_root_cv: dict[int, np.ndarray] = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        poses, detections, pose_scores = detect_marker_poses(
            frame=frame,
            refs=refs,
            ref_bits=ref_bits,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            marker_points=marker_points,
            mismatch_threshold=args.match_threshold,
        )

        for det in detections:
            c = det["corners"].astype(np.int32).reshape(-1, 1, 2)
            marker_id = det["id"]
            score = det["score"]
            col = (30, 220, 30) if score < 0.14 else (0, 180, 255)
            cv2.polylines(frame, [c], True, col, 2, cv2.LINE_AA)
            cx = int(np.mean(c[:, 0, 0]))
            cy = int(np.mean(c[:, 0, 1]))
            cv2.putText(
                frame,
                f"id {marker_id} ({score:.2f})",
                (cx - 42, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                col,
                2,
                cv2.LINE_AA,
            )

        visible = sorted(poses.keys())
        visible_cam_mats = camera_marker_mats_from_poses(poses)
        if ORIGIN_ID in visible_cam_mats and ORIGIN_ID not in known_root_cv:
            known_root_cv[ORIGIN_ID] = np.eye(4, dtype=np.float64)

        root_estimates_cv = estimate_root_mats_from_visible(visible_cam_mats, known_root_cv)
        if root_estimates_cv:
            for marker_id, t_root_marker_cv in root_estimates_cv.items():
                if marker_id == ORIGIN_ID:
                    known_root_cv[ORIGIN_ID] = np.eye(4, dtype=np.float64)
                elif marker_id not in known_root_cv:
                    known_root_cv[marker_id] = t_root_marker_cv
                else:
                    # Slow refinement to reduce drift while keeping continuity.
                    known_root_cv[marker_id] = mean_pose([known_root_cv[marker_id], t_root_marker_cv])

                mat = apply_basis_opencv_to_three(t_root_marker_cv)
                if marker_id == ORIGIN_ID:
                    score_ok = True
                else:
                    s = pose_scores.get(marker_id, 999.0)
                    score_ok = s <= args.sample_score_max
                if not score_ok:
                    continue
                bucket = samples.setdefault(marker_id, [])
                bucket.append(mat)
                if len(bucket) > max_samples_per_marker:
                    bucket.pop(0)
                if marker_id != ORIGIN_ID:
                    sb = sample_scores.setdefault(marker_id, [])
                    sb.append(float(pose_scores.get(marker_id, 999.0)))
                    if len(sb) > max_samples_per_marker:
                        sb.pop(0)

        sample_counts = {k: len(v) for k, v in samples.items() if len(v) > 0}
        cv2.putText(
            frame,
            f"Visible: {visible} | Root seeded: {ORIGIN_ID in known_root_cv} | Need origin: {ORIGIN_ID}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if ORIGIN_ID in known_root_cv else (0, 140, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Samples: {sample_counts}",
            (20, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 220, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("Psyche AR Calibration", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            try:
                save_config(
                    output_path=output_path,
                    samples=samples,
                    sample_scores=sample_scores,
                    min_samples=args.min_samples,
                    max_median_score=args.max_median_score,
                    target_ids=target_ids,
                    merge_existing=args.merge_existing,
                    min_surface_y=args.min_surface_y,
                )
            except Exception as exc:
                print(f"Auto-save on quit failed: {exc}")
            break
        if key == ord("c"):
            samples = {i: [] for i in sorted(ALL_IDS)}
            sample_scores = {i: [] for i in sorted(ALL_IDS)}
            known_root_cv = {}
            print("Cleared samples.")
        if key == ord("s"):
            try:
                save_config(
                    output_path=output_path,
                    samples=samples,
                    sample_scores=sample_scores,
                    min_samples=args.min_samples,
                    max_median_score=args.max_median_score,
                    target_ids=target_ids,
                    merge_existing=args.merge_existing,
                    min_surface_y=args.min_surface_y,
                )
            except Exception as exc:
                print(f"Save failed: {exc}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
