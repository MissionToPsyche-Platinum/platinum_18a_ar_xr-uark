import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from calibration import (
    ORIGIN_ID,
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
        description="Live probe for table-marker rotation (IDs 1,3,4,6) and center stability."
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
        default=180,
        help="Rolling sample count kept per marker for smoothing",
    )
    parser.add_argument(
        "--report-out",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "public" / "table_rotation_report.json"),
        help="JSON report output path (used with key [j])",
    )
    return parser.parse_args()


def rotation_to_yaw_pitch_roll_deg(r: np.ndarray) -> tuple[float, float, float]:
    """
    Approximate Y-up intrinsic YXZ decomposition in degrees.
    Useful for debugging relative marker heading/tilt consistency.
    """
    pitch = float(np.degrees(np.arcsin(np.clip(-r[1, 2], -1.0, 1.0))))
    yaw = float(np.degrees(np.arctan2(r[0, 2], r[2, 2])))
    roll = float(np.degrees(np.arctan2(r[1, 0], r[1, 1])))
    return yaw, pitch, roll


def build_flattened_table_solution(samples_three: dict[int, list[np.ndarray]]) -> tuple[dict[int, np.ndarray], np.ndarray | None]:
    averaged: dict[int, np.ndarray] = {}
    for marker_id in sorted(TABLE_IDS):
        mats = samples_three.get(marker_id, [])
        if mats:
            averaged[int(marker_id)] = mean_pose(mats)

    if len(averaged) < 3:
        return averaged, None

    pts = [averaged[mid][:3, 3] for mid in sorted(averaged.keys())]
    n, c = fit_plane(pts)
    r_align = rotation_align_a_to_b(n, np.array([0.0, 1.0, 0.0], dtype=np.float64))

    for mid in list(averaged.keys()):
        m = averaged[mid].copy()
        m[:3, 3] = r_align @ m[:3, 3]
        m[:3, :3] = orthonormalize_rotation(r_align @ m[:3, :3])
        averaged[mid] = m

    c_aligned = r_align @ c
    up_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    for mid in list(averaged.keys()):
        m = averaged[mid].copy()
        m[:3, 3] = project_point_to_plane(m[:3, 3], up_y, c_aligned)
        m[:3, 3][1] = 0.0
        m = align_table_rotation_to_plane(m, up_y)
        m[:3, :3] = orthonormalize_rotation(m[:3, :3])
        averaged[mid] = m

    return averaged, c_aligned


def summarize(samples_three: dict[int, list[np.ndarray]]) -> tuple[dict[int, dict], list[float] | None]:
    flat_mats, _ = build_flattened_table_solution(samples_three)
    per_marker: dict[int, dict] = {}
    table_positions = []
    for marker_id in sorted(flat_mats.keys()):
        mats = samples_three.get(marker_id, [])
        m = flat_mats[marker_id]
        if not mats:
            continue
        t = m[:3, 3]
        yaw, pitch, roll = rotation_to_yaw_pitch_roll_deg(m[:3, :3])
        per_marker[int(marker_id)] = {
            "samples": len(mats),
            "translation_m": {"x": float(t[0]), "y": float(t[1]), "z": float(t[2])},
            "rotation_deg": {"yaw": yaw, "pitch": pitch, "roll": roll},
            "poseMatrix": flatten_for_three(m),
        }
        table_positions.append(t)

    center = None
    if table_positions:
        c = np.mean(np.array(table_positions, dtype=np.float64), axis=0)
        center = [float(c[0]), float(c[1]), float(c[2])]
    return per_marker, center


def print_report(samples_three: dict[int, list[np.ndarray]]) -> None:
    per_marker, center = summarize(samples_three)
    print("---- Table Rotation Probe ----")
    if not per_marker:
        print("No table marker samples yet.")
        return
    for marker_id in sorted(per_marker.keys()):
        row = per_marker[marker_id]
        r = row["rotation_deg"]
        t = row["translation_m"]
        print(
            f"id {marker_id}: samples={row['samples']}, "
            f"yaw={r['yaw']:.2f}, pitch={r['pitch']:.2f}, roll={r['roll']:.2f}, "
            f"t=({t['x']:.4f}, {t['y']:.4f}, {t['z']:.4f})"
        )
    if center is not None:
        print(f"center(1,3,4,6) = ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")


def write_report(samples_three: dict[int, list[np.ndarray]], report_path: Path) -> None:
    per_marker, center = summarize(samples_three)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "frame": "three.js Y-up; table plane flattened to Y=0",
        "table_marker_ids": sorted(int(i) for i in TABLE_IDS),
        "origin_marker_id": int(ORIGIN_ID),
        "markers": per_marker,
        "center_1346_m": {"x": center[0], "y": center[1], "z": center[2]} if center else None,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote report: {report_path}")


def main():
    args = parse_args()
    report_path = Path(args.report_out)
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

    samples_three: dict[int, list[np.ndarray]] = {int(i): [] for i in sorted(TABLE_IDS)}
    known_root_cv: dict[int, np.ndarray] = {}

    print("Table rotation probe started.")
    print("Controls: [p] print report, [j] write JSON report, [c] clear samples, [q] save+quit")
    print(f"Camera index: {args.camera}")
    print(f"Tracking marker IDs: {sorted(TABLE_IDS)} (origin marker: {ORIGIN_ID})")

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
            color = (20, 220, 20) if marker_id in TABLE_IDS else (0, 180, 255)
            cv2.polylines(frame, [corners], True, color, 2, cv2.LINE_AA)
            cx = int(np.mean(corners[:, 0, 0]))
            cy = int(np.mean(corners[:, 0, 1]))
            cv2.putText(
                frame,
                f"id {marker_id}",
                (cx - 26, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

        visible_cam_mats = camera_marker_mats_from_poses(poses)
        if ORIGIN_ID in visible_cam_mats:
            known_root_cv[ORIGIN_ID] = np.eye(4, dtype=np.float64)

        root_estimates_cv = estimate_root_mats_from_visible(visible_cam_mats, known_root_cv)
        for marker_id, t_root_marker_cv in root_estimates_cv.items():
            if marker_id == ORIGIN_ID:
                known_root_cv[ORIGIN_ID] = np.eye(4, dtype=np.float64)
            elif marker_id not in known_root_cv:
                known_root_cv[marker_id] = t_root_marker_cv
            else:
                known_root_cv[marker_id] = mean_pose([known_root_cv[marker_id], t_root_marker_cv])

            if marker_id not in TABLE_IDS:
                continue
            m_three = apply_basis_opencv_to_three(t_root_marker_cv)
            bucket = samples_three[int(marker_id)]
            bucket.append(m_three)
            if len(bucket) > args.max_samples:
                bucket.pop(0)

        y = 30
        for marker_id in sorted(TABLE_IDS):
            mats = samples_three.get(int(marker_id), [])
            if not mats:
                text = f"id {marker_id}: waiting..."
            else:
                flat_mats, _ = build_flattened_table_solution(samples_three)
                show = flat_mats.get(marker_id, mean_pose(mats))
                yaw, pitch, roll = rotation_to_yaw_pitch_roll_deg(show[:3, :3])
                text = (
                    f"id {marker_id}: yaw {yaw:6.1f}  pitch {pitch:6.1f}  "
                    f"roll {roll:6.1f}  n={len(mats):3d}"
                )
            cv2.putText(frame, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 255), 1, cv2.LINE_AA)
            y += 22

        _, center = summarize(samples_three)
        if center is not None:
            cv2.putText(
                frame,
                f"center(1,3,4,6): x={center[0]:.4f}, y={center[1]:.4f}, z={center[2]:.4f}",
                (18, y + 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (90, 255, 180),
                1,
                cv2.LINE_AA,
            )

        cv2.imshow("Psyche AR - Table Rotation Probe", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            try:
                write_report(samples_three, report_path)
            except Exception as exc:
                print(f"Auto-save on quit failed: {exc}")
            break
        if key == ord("p"):
            print_report(samples_three)
        if key == ord("j"):
            write_report(samples_three, report_path)
        if key == ord("c"):
            samples_three = {int(i): [] for i in sorted(TABLE_IDS)}
            known_root_cv = {}
            print("Cleared samples.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
