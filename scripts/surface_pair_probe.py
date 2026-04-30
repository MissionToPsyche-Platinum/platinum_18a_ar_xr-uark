import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from calibration import (
    ORIGIN_ID,
    SURFACE_IDS,
    TABLE_IDS,
    align_table_rotation_to_plane,
    apply_basis_opencv_to_three,
    build_reference_bits,
    camera_marker_mats_from_poses,
    detect_marker_poses,
    estimate_root_mats_from_visible,
    fit_plane,
    flatten_for_three,
    load_reference_markers,
    mean_pose,
    orthonormalize_rotation,
    project_point_to_plane,
    rotation_align_a_to_b,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Live probe for calibrating surface markers (0,2,5,7) via table markers (1,3,4,6) in any pairing order."
    )
    parser.add_argument("--camera", type=int, default=1, help="Camera index (default: 1)")
    parser.add_argument(
        "--marker-length",
        type=float,
        default=0.0508,
        help="Printed marker black-square width in meters (default: 0.0508 for 2-inch prints)",
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
        help="Max binary mismatch ratio for marker match acceptance",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=240,
        help="Rolling sample count kept per marker for smoothing",
    )
    parser.add_argument(
        "--table-report-in",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "public" / "table_rotation_report.json"),
        help="Optional input table report used to seed known table poses",
    )
    parser.add_argument(
        "--unlock-table",
        action="store_true",
        help="Allow live refinement of table markers (default keeps 1,3,4,6 locked from table report)",
    )
    parser.add_argument(
        "--report-out",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "public" / "surface_pair_report.json"),
        help="JSON report output path (used with key [j])",
    )
    parser.add_argument(
        "--min-surface-y",
        type=float,
        default=0.005,
        help="Minimum positive Y for exported surface marker translations",
    )
    return parser.parse_args()


def rotation_to_yaw_pitch_roll_deg(r: np.ndarray) -> tuple[float, float, float]:
    pitch = float(np.degrees(np.arcsin(np.clip(-r[1, 2], -1.0, 1.0))))
    yaw = float(np.degrees(np.arctan2(r[0, 2], r[2, 2])))
    roll = float(np.degrees(np.arctan2(r[1, 0], r[1, 1])))
    return yaw, pitch, roll


def basis_three_to_opencv(m_three: np.ndarray) -> np.ndarray:
    b = np.eye(4, dtype=np.float64)
    b[1, 1] = -1.0
    b[2, 2] = -1.0
    return b @ m_three @ b


def parse_pose_matrix_three(flat: list[float]) -> np.ndarray | None:
    if not isinstance(flat, list) or len(flat) < 16:
        return None
    arr = np.array(flat[:16], dtype=np.float64)
    return arr.reshape((4, 4)).T


def load_seed_table_root_cv(path: Path) -> dict[int, np.ndarray]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    markers = data.get("markers", {})
    if not isinstance(markers, dict):
        return {}
    out: dict[int, np.ndarray] = {}
    for marker_id in sorted(TABLE_IDS):
        row = markers.get(str(marker_id))
        if not isinstance(row, dict):
            continue
        m_three = parse_pose_matrix_three(row.get("poseMatrix"))
        if m_three is None:
            continue
        out[int(marker_id)] = basis_three_to_opencv(m_three)
    return out


def build_flattened_solution(
    samples_three: dict[int, list[np.ndarray]],
    seed_table_three: dict[int, np.ndarray],
    lock_table: bool,
) -> tuple[dict[int, np.ndarray], np.ndarray | None]:
    averaged: dict[int, np.ndarray] = {}

    if lock_table:
        for marker_id in sorted(TABLE_IDS):
            m = seed_table_three.get(int(marker_id))
            if m is not None:
                averaged[int(marker_id)] = m.copy()

    for marker_id, mats in samples_three.items():
        if mats:
            if lock_table and marker_id in TABLE_IDS and marker_id in averaged:
                continue
            averaged[int(marker_id)] = mean_pose(mats)

    table_present = [mid for mid in sorted(TABLE_IDS) if mid in averaged]
    if len(table_present) < 3:
        return averaged, None

    pts = [averaged[mid][:3, 3] for mid in table_present]
    n, c = fit_plane(pts)
    r_align = rotation_align_a_to_b(n, np.array([0.0, 1.0, 0.0], dtype=np.float64))

    for mid in list(averaged.keys()):
        m = averaged[mid].copy()
        m[:3, 3] = r_align @ m[:3, 3]
        m[:3, :3] = orthonormalize_rotation(r_align @ m[:3, :3])
        averaged[mid] = m

    c_aligned = r_align @ c
    up_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    # Pin table markers to Y=0 and flatten their tilt.
    for mid in table_present:
        m = averaged[mid].copy()
        m[:3, 3] = project_point_to_plane(m[:3, 3], up_y, c_aligned)
        m[:3, 3][1] = 0.0
        m = align_table_rotation_to_plane(m, up_y)
        m[:3, :3] = orthonormalize_rotation(m[:3, :3])
        averaged[mid] = m

    return averaged, c_aligned


def summarize(
    samples_three: dict[int, list[np.ndarray]],
    seed_table_three: dict[int, np.ndarray],
    lock_table: bool,
    min_surface_y: float,
) -> tuple[dict[int, dict], list[float] | None]:
    solved, _ = build_flattened_solution(samples_three, seed_table_three=seed_table_three, lock_table=lock_table)
    out: dict[int, dict] = {}
    table_positions = []

    for marker_id in sorted(solved.keys()):
        m = solved[marker_id].copy()
        # Surface markers should remain above table plane (positive Y).
        if marker_id in SURFACE_IDS:
            m[1, 3] = max(abs(float(m[1, 3])), float(min_surface_y))

        t = m[:3, 3]
        yaw, pitch, roll = rotation_to_yaw_pitch_roll_deg(m[:3, :3])
        out[int(marker_id)] = {
            "group": "table" if marker_id in TABLE_IDS else ("surface" if marker_id in SURFACE_IDS else "other"),
            "samples": len(samples_three.get(marker_id, [])),
            "translation_m": {"x": float(t[0]), "y": float(t[1]), "z": float(t[2])},
            "rotation_deg": {"yaw": yaw, "pitch": pitch, "roll": roll},
            "poseMatrix": flatten_for_three(m),
        }
        if marker_id in TABLE_IDS:
            table_positions.append(t)

    center = None
    if table_positions:
        c = np.mean(np.array(table_positions, dtype=np.float64), axis=0)
        center = [float(c[0]), float(c[1]), float(c[2])]
    return out, center


def print_report(
    samples_three: dict[int, list[np.ndarray]],
    seed_table_three: dict[int, np.ndarray],
    lock_table: bool,
    min_surface_y: float,
) -> None:
    markers, center = summarize(
        samples_three,
        seed_table_three=seed_table_three,
        lock_table=lock_table,
        min_surface_y=min_surface_y,
    )
    print("---- Surface Pair Probe ----")
    if not markers:
        print("No samples yet.")
        return
    for marker_id in sorted(markers.keys()):
        row = markers[marker_id]
        r = row["rotation_deg"]
        t = row["translation_m"]
        print(
            f"id {marker_id} [{row['group']}]: n={row['samples']}, "
            f"yaw={r['yaw']:.2f}, pitch={r['pitch']:.2f}, roll={r['roll']:.2f}, "
            f"t=({t['x']:.4f}, {t['y']:.4f}, {t['z']:.4f})"
        )
    if center is not None:
        print(f"center(1,3,4,6) = ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")


def write_report(
    samples_three: dict[int, list[np.ndarray]],
    seed_table_three: dict[int, np.ndarray],
    lock_table: bool,
    report_path: Path,
    min_surface_y: float,
) -> None:
    markers, center = summarize(
        samples_three,
        seed_table_three=seed_table_three,
        lock_table=lock_table,
        min_surface_y=min_surface_y,
    )
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "frame": "three.js Y-up; table flattened to Y=0; surface markers constrained to +Y",
        "table_lock_mode": "locked_from_table_report" if lock_table else "live_refined",
        "origin_marker_id": int(ORIGIN_ID),
        "table_marker_ids": sorted(int(i) for i in TABLE_IDS),
        "surface_marker_ids": sorted(int(i) for i in SURFACE_IDS),
        "markers": markers,
        "center_1346_m": {"x": center[0], "y": center[1], "z": center[2]} if center else None,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote report: {report_path}")


def main():
    args = parse_args()
    report_path = Path(args.report_out)
    seed_path = Path(args.table_report_in)

    refs = load_reference_markers(refs_dir=Path(args.refs_dir), size=160)
    ref_bits = build_reference_bits(refs)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    fx, fy, cx, cy = 1000.0, 1000.0, 960.0, 540.0
    camera_matrix = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    marker_points = np.array(
        [
            [-args.marker_length / 2, args.marker_length / 2, 0],
            [args.marker_length / 2, args.marker_length / 2, 0],
            [args.marker_length / 2, -args.marker_length / 2, 0],
            [-args.marker_length / 2, -args.marker_length / 2, 0],
        ],
        dtype=np.float32,
    )

    tracked_ids = sorted(int(i) for i in (TABLE_IDS | SURFACE_IDS))
    samples_three: dict[int, list[np.ndarray]] = {mid: [] for mid in tracked_ids}
    known_root_cv: dict[int, np.ndarray] = load_seed_table_root_cv(seed_path)
    if ORIGIN_ID not in known_root_cv:
        known_root_cv[ORIGIN_ID] = np.eye(4, dtype=np.float64)
    lock_table = not args.unlock_table
    if lock_table and len([k for k in known_root_cv.keys() if k in TABLE_IDS]) < 3:
        raise RuntimeError(
            f"Table lock mode requires seeded table poses in {seed_path}. "
            "Run probe:table-rotation first or pass --unlock-table."
        )
    seed_table_three: dict[int, np.ndarray] = {}
    for marker_id in TABLE_IDS:
        m = known_root_cv.get(int(marker_id))
        if m is not None:
            seed_table_three[int(marker_id)] = apply_basis_opencv_to_three(m)

    print("Surface pair probe started.")
    print("Controls: [p] print report, [j] write JSON report, [c] clear samples, [q] save+quit")
    print(f"Camera index: {args.camera}")
    print(f"Tracking table IDs: {sorted(TABLE_IDS)}")
    print(f"Tracking surface IDs: {sorted(SURFACE_IDS)}")
    print(f"Seed table report: {seed_path} (loaded={seed_path.exists()})")
    print(f"Table lock mode: {lock_table}")
    print("Pairing logic: any visible known anchor can calibrate any visible target marker.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        poses, detections, _ = detect_marker_poses(
            frame=frame,
            refs=refs,
            ref_bits=ref_bits,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            marker_points=marker_points,
            mismatch_threshold=args.match_threshold,
        )

        for det in detections:
            marker_id = int(det["id"])
            corners = det["corners"].astype(np.int32).reshape(-1, 1, 2)
            if marker_id in TABLE_IDS:
                color = (20, 220, 20)
            elif marker_id in SURFACE_IDS:
                color = (40, 170, 255)
            else:
                color = (130, 130, 130)
            cv2.polylines(frame, [corners], True, color, 2, cv2.LINE_AA)
            cx = int(np.mean(corners[:, 0, 0]))
            cy = int(np.mean(corners[:, 0, 1]))
            cv2.putText(frame, f"id {marker_id}", (cx - 22, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)

        visible_cam_mats = camera_marker_mats_from_poses(poses)
        if ORIGIN_ID in visible_cam_mats:
            known_root_cv[ORIGIN_ID] = np.eye(4, dtype=np.float64)

        root_estimates_cv = estimate_root_mats_from_visible(visible_cam_mats, known_root_cv)
        for marker_id, t_root_marker_cv in root_estimates_cv.items():
            if marker_id == ORIGIN_ID:
                known_root_cv[ORIGIN_ID] = np.eye(4, dtype=np.float64)
            elif lock_table and marker_id in TABLE_IDS and marker_id in known_root_cv:
                # Keep table frame fixed from seeded report.
                pass
            elif marker_id not in known_root_cv:
                known_root_cv[marker_id] = t_root_marker_cv
            else:
                known_root_cv[marker_id] = mean_pose([known_root_cv[marker_id], t_root_marker_cv])

            if marker_id not in samples_three:
                continue
            if lock_table and marker_id in TABLE_IDS:
                continue
            m_three = apply_basis_opencv_to_three(t_root_marker_cv)
            bucket = samples_three[int(marker_id)]
            bucket.append(m_three)
            if len(bucket) > args.max_samples:
                bucket.pop(0)

        markers, center = summarize(
            samples_three,
            seed_table_three=seed_table_three,
            lock_table=lock_table,
            min_surface_y=args.min_surface_y,
        )

        y = 28
        for marker_id in tracked_ids:
            row = markers.get(marker_id)
            if row is None:
                text = f"id {marker_id}: waiting..."
            else:
                r = row["rotation_deg"]
                t = row["translation_m"]
                text = (
                    f"id {marker_id} [{row['group']}]: y={t['y']:+.3f} "
                    f"yaw={r['yaw']:+5.1f} n={row['samples']:3d}"
                )
            cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (230, 230, 255), 1, cv2.LINE_AA)
            y += 21

        if center is not None:
            cv2.putText(
                frame,
                f"center(1,3,4,6): x={center[0]:.4f}, y={center[1]:.4f}, z={center[2]:.4f}",
                (16, y + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                (90, 255, 180),
                1,
                cv2.LINE_AA,
            )

        visible_sorted = sorted(int(k) for k in poses.keys())
        cv2.putText(
            frame,
            f"Visible: {visible_sorted}",
            (16, frame.shape[0] - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (220, 220, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("Psyche AR - Surface Pair Probe", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            try:
                write_report(
                    samples_three,
                    seed_table_three=seed_table_three,
                    lock_table=lock_table,
                    report_path=report_path,
                    min_surface_y=args.min_surface_y,
                )
            except Exception as exc:
                print(f"Auto-save on quit failed: {exc}")
            break
        if key == ord("p"):
            print_report(
                samples_three,
                seed_table_three=seed_table_three,
                lock_table=lock_table,
                min_surface_y=args.min_surface_y,
            )
        if key == ord("j"):
            write_report(
                samples_three,
                seed_table_three=seed_table_three,
                lock_table=lock_table,
                report_path=report_path,
                min_surface_y=args.min_surface_y,
            )
        if key == ord("c"):
            samples_three = {mid: [] for mid in tracked_ids}
            known_root_cv = load_seed_table_root_cv(seed_path)
            if ORIGIN_ID not in known_root_cv:
                known_root_cv[ORIGIN_ID] = np.eye(4, dtype=np.float64)
            seed_table_three = {}
            for marker_id in TABLE_IDS:
                m = known_root_cv.get(int(marker_id))
                if m is not None:
                    seed_table_three[int(marker_id)] = apply_basis_opencv_to_three(m)
            print("Cleared samples.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
