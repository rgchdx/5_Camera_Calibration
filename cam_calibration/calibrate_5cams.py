#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5-Camera Calibration Script (OpenCV)
- Intrinsic per camera
- Multi-view extrinsics relative to a reference camera (cam0)
- Supports checkerboard or ChArUco, pinohle or fisheye models
- Outputs YAML with all parameters
"""
import os
import glob
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import cv2
from collections import defaultdict

try:
    import yaml # for YAML output
    HAVE_YAML = True
except Exception:
    HAVE_YAML = False

# ----------------------------------
# Utilities
# ----------------------------------
@dataclass                  # Data class for camera calibration and eliminating __init__ boilerplate
class cameraCalib:
    K: np.ndarray           # 3x3 Intrinsic matrix
    dist: np.ndarray        # (k, ) shape: Distortion coefficients
    res: Tuple[int, int]    # (w, h): Resolution
    model: str              # 'pinhole' or 'fisheye' Camera model

def detect_checkerboard(gray, rows, cols):
    # rows x cols = inner corners
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
    if not ret:
        return False, None
    # refine corner locations
    corners = cv2.cornerSubPix(
        gray, corners, winSize=(11, 11), zeroZone=(-1, -1),
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    )
    return True, corners

def make_charuco_board(rows, cols, square, marker_len, dict_name):
    # rows, cols are number of squares (not inner corners)
    d = getattr(cv2.aruco, dict_name)
    aruco_dict = cv2.aruco.getPredefinedDictionary(d)
    board = cv2.aruco.CharucoBoard((cols, rows), square, marker_len, aruco_dict)
    return aruco_dict, board

def detect_charuco(gray, aruco_dict, board):
    params = cv2.aruco.DetectorParameters()
    print("params:", params)
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    print("detector:", detector)
    corners, ids, _ = detector.detectMarkers(gray)
    print("corners:", corners, "ids:", ids)
    if ids is None or len(ids) == 0:
        return False, None, None
    _, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
    if ch_ids is None or len(ch_ids) < 6:
        return False, None, None
    return True, ch_corners, ch_ids

def objpoints_checkerboard(rows, cols, square):
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    return objp # (N, 3) array

def intrinsic_calibration(image_paths: List[str], pattern: str, rows: int, cols: int,
                          square: float, model: str, charuco_marker: float = None,
                          dict_name: str = "DICT_4X4_50"):
    objpoints = []
    imgpoints = []
    im_size = None
    if pattern == "charuco":
        aruco_dict, board = make_charuco_board(rows, cols, square, charuco_marker, dict_name)
    for p in image_paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Unable to load image {p}, skipping.")
            continue
        if im_size is None:
            im_size = (img.shape[1], img.shape[0])  # (w, h)
        if pattern == "checkerboard":
            found, corners = detect_checkerboard(img, rows, cols, square)
            if not found:
                print(f"Checkerboard not found in image {p}, skipping.")
                continue
            objp = objpoints_checkerboard(rows, cols, square)
            objpoints.append(objp)
            imgpoints.append(corners)
        else:
            found, ch_corners, ch_ids = detect_charuco(img, aruco_dict, board)
            if not found:
                print(f"ChArUco board not found in image {p}, skipping.")
                continue
            # cv2.aruco.calibrateCameraCharuco expects all frames aggregated
            # We'll collect per-frame and call special function below
            # For intrinsics, we can use calibrateCameraCharuco
            # Build a one-frame dataset:
            if 'all_corners' not in locals():
                all_corners, all_ids,  imsize = [], [], im_size
            all_corners.append(ch_corners)
            all_ids.append(ch_ids)
    if im_size is None:
        raise RuntimeError("No readable images found for calibration.")
    if pattern == "checkerboard":
        if model == "pinhole":
            rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, im_size, None, None,
                flags = cv2.CALIB_RATIONAL_MODEL
            )
        else:  # fisheye
            K = np.eye(3)
            D = np.zeros((4, 1))
            flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
            rms, _, _, _, _ = cv2.fisheye.calibrate(
                [op.reshape(-1, 1, 3) for op in objpoints],
                [ip for ip in imgpoints],
                im_size, K, D, None, None, flags=flags,
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
            )
            dist = D
    else:  # charuco
        # ChArUco intrinsics (pinhole only, fisheye not supported directly by aruco extension)
        # Aggregate from locals
        # ChArUco intrinsics (pinhole only)
        if 'all_corners' not in locals() or len(all_corners) == 0:
            print(f"[WARN] No ChArUco boards detected for this camera. Skipping calibration.")
            return None, None

        rms, K, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            charucoCorners=all_corners,
            charucoIds=all_ids,
            board=board,
            imageSize=im_size,
            cameraMatrix=None,
            distCoeffs=None
        )
        if model == "fisheye":
            raise ValueError("Fisheye model not supported for ChArUco calibration in this implementation.")
    return rms, cameraCalib(K=K, dist=dist, res=im_size, model=model)   # potential error here

def estimate_board_pose(gray, pattern, rows, cols, square, K, dist, charuco_marker=None, dict_name = "DICT_4X4_50"):
    if pattern == "checkerboard":
        found, corners = detect_checkerboard(gray, rows, cols)
        if not found:
            return False, None, None
        objp = objpoints_checkerboard(rows, cols, square)
        ret, rvec, tvec = cv2.solvePnP(objp, corners, K, dist)
        return True, rvec, tvec
    else:
        aruco_dict, board = make_charuco_board(rows, cols, square, charuco_marker, dict_name)
        found, ch_corners, ch_ids = detect_charuco(gray, aruco_dict, board)
        if not found:
            return False, None, None
        ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(ch_corners, ch_ids, board, K, dist, None, None)
        if not ret:
            return False, None, None
        return True, rvec, tvec

def rvec_tvec_to_rt(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3, 1)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3:4] = t
    return T

def invert_T(T):
    R = T[:3, :3]
    t = T[:3, 3:4]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3:4] = -R.T @ t
    return Ti

def extrinsics_from_shared_board(
    data_root,
    cams,
    pattern,
    rows,
    cols,
    square,
    calibs: Dict[str, cameraCalib],
    charuco_marker=None,
    dict_name="DICT_4X4_50"
):
    """
    For frames where multiple cameras see the board at (roughly) the same time,
    we estimate each cam's pose wrt the board, then compute relative transforms.
    cam_i -> cam0 by T_i0 = T_i_board * inv(T_0_board)
    """
    def se3_log(T):
        R = T[:3, :3]
        t = T[:3, 3:4]
        rvec, _ = cv2.Rodrigues(R)
        return np.concatenate([rvec.flatten(), t.flatten()])

    def se3_exp(xi):
        rvec = xi[:3].reshape(3, 1)
        t = xi[3:].reshape(3, 1)
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3:4] = t
        return T

    # --- initialize outputs ---
    ref = cams[0]
    extrinsics = {ref: np.eye(4)}

    # --- collect per-camera image lists ---
    cam_files = {}
    for c in cams:
        # Support both:  (1) each camera has its own subfolder, or (2) all images in one folder
        folder = os.path.join(data_root, c) if os.path.isdir(os.path.join(data_root, c)) else data_root
        files = sorted(glob.glob(os.path.join(folder, "*.jpg")) +
                       glob.glob(os.path.join(folder, "*.png")))
        cam_files[c] = files
        print(f"[INFO] Found {len(files)} images for camera '{c}' in {folder}")

    # --- ensure there are images ---
    if any(len(v) == 0 for v in cam_files.values()):
        print("[WARN] One or more cameras have no images. Skipping extrinsic estimation.")
        return ref, extrinsics

    min_len = min(len(v) for v in cam_files.values())
    print(f"[INFO] Using {min_len} frames per camera for synchronization")

    frame_to_Tc_board = defaultdict(dict)

    for idx in range(min_len):
        for c in cams:
            p = cam_files[c][idx]
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[WARN] Could not read {p}")
                continue

            K = calibs[c].K
            dist = calibs[c].dist
            found, rvec, tvec = estimate_board_pose(
                img, pattern, rows, cols, square, K, dist,
                charuco_marker=charuco_marker, dict_name=dict_name
            )
            if not found:
                continue

            T_c_board = rvec_tvec_to_rt(rvec, tvec)
            frame_to_Tc_board[idx][c] = T_c_board

    if len(frame_to_Tc_board) == 0:
        print("[ERROR] No valid board detections found across cameras.")
        return ref, extrinsics

    # --- compute relative transforms ---
    rel_lists = {c: [] for c in cams if c != ref}
    for idx, d in frame_to_Tc_board.items():
        if ref not in d:
            continue
        T_ref_board = d[ref]
        T_board_ref = invert_T(T_ref_board)
        for c, T_c_board in d.items():
            if c == ref:
                continue
            T_c_ref = T_c_board @ T_board_ref
            rel_lists[c].append(T_c_ref)

    # --- average the relative transforms ---
    for c in cams:
        if c == ref:
            continue
        Ts = rel_lists[c]
        if len(Ts) == 0:
            print(f"[WARN] No shared board views found between {ref} and {c}")
            continue
        xis = np.stack([se3_log(T) for T in Ts], axis=0)
        median_xi = np.median(xis, axis=0)
        extrinsics[c] = se3_exp(median_xi)

    print(f"[INFO] Extrinsic estimation complete. Found {len(extrinsics)} camera transforms.")
    return ref, extrinsics

def save_yaml_json(out_path, cams, calibs: Dict[str, cameraCalib], ref_cam: str, extrinsics: Dict[str, np.ndarray]):
    data = {
        "reference_camera": ref_cam,
        "cameras": {}
    }
    for c in cams:
        cc = calibs[c]
        entry = {
            "model": cc.model,
            "resolution": {"width": int(cc.res[0]), "height": int(cc.res[1])},
            "K": cc.K.tolist(),
            "dist": cc.dist.flatten().tolist(),
            "T_cam_ref": extrinsics[c].tolist() if c != ref_cam else np.eye(4).tolist()
        }
        data["cameras"][c] = entry
    ext = os.path.splitext(out_path)[1].lower()
    if ext in [".yaml", ".yml"]:
        with open(out_path, "w") as f:
            yaml.safe_dump(data, f, sort_keys = False)
    else: 
        with open(out_path, "w") as f:
            json.dump(data, f, indent = 2)
    
    # define main here later after all testing is done.

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_root", type=str, required = True, 
        help="Root folder containing per-camera subfolders or all images"
    )
    ap.add_argument(
        "--cams", nargs="+", required=True,
        help="Camera folder names, e.g., cam0 cam1 cam2 cam3 cam4"
    )
    ap.add_argument(
        "--pattern", choices=["checkerboard", "charuco"], default="checkerboard",
        help="Calibration pattern type"
    )
    ap.add_argument(
        "--rows", type=int, required=True,
        help="Checkerboard inner rows OR ChArUco rows (squares)"
    )
    ap.add_argument(
        "--cols", type=int, required=True,
        help="Checkerboard inner cols OR ChArUco cols (squares)"
    )
    ap.add_argument(
        "--square", type=float, required=True,
        help="Square size in meters"
    )
    ap.add_argument(
        "--marker", type=float, default=None,
        help="ChArUco marker size in meters (required if charuco)"
    )
    ap.add_argument(
        "--dict", type=str, default="DICT_4X4_50",
        help="ArUco dictionary name for ChArUco"
    )
    ap.add_argument(
        "--model", choices=["pinhole", "fisheye"], default="pinhole",
        help="Camera model type"
    )
    ap.add_argument(
        "--output", required=True,
        help="Output YAML/JSON file path"
    )
    args = ap.parse_args()
    cams = args.cams
    # --- 1) Intrinsic Calibration ---
    calibs: Dict[str, cameraCalib] = {}
    for c in cams:
        # Gather images
        paths = sorted(
            glob.glob(os.path.join(args.data_root, c, "*.jpg")) +
            glob.glob(os.path.join(args.data_root, c, "*.png"))
        )
        if len(paths) < 10:
            print(f"[WARN] {c}: only {len(paths)} images found; results may be poor.")
        print(f"[INFO] Calibrating intrinsics for {c} with {len(paths)} images...")
        if args.pattern == "charuco" and args.marker is None:
            raise ValueError("For ChArUco, please provide --marker (marker length in meters).")
        rms, calib = intrinsic_calibration(
            image_paths=paths,
            pattern=args.pattern,
            rows=args.rows,
            cols=args.cols,
            square=args.square,
            model=args.model,
            charuco_marker=args.marker,
            dict_name=args.dict
        )

        # Skip if no calibration result
        if rms is None or calib is None:
            print(f"[WARN] {c}: Calibration skipped (no valid ChArUco detections).")
            continue

        print(f"[INFO] {c}: RMS reprojection error = {rms:.4f} px")

    # --- 2) Extrinsic Calibration ---
    print("[INFO] Estimating extrinsics via multi-view shared board observations...")
    ref_cam, extrinsics = extrinsics_from_shared_board(
        data_root=args.data_root,
        cams=cams,
        pattern=args.pattern,
        rows=args.rows,
        cols=args.cols,
        square=args.square,
        calibs=calibs,
        charuco_marker=args.marker,
        dict_name=args.dict
    )
    # Ensure reference camera has identity transform
    extrinsics[ref_cam] = np.eye(4)

    # --- 3) Save results ---
    print(f"[INFO] Saving calibration results to {args.output} ...")
    save_yaml_json(args.output, cams, calibs, ref_cam, extrinsics)

    print("[DONE] Calibration completed successfully.")

if __name__ == "__main__":
    main()