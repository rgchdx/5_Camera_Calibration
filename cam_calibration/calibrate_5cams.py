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
        rms, K, dist, rvecs, tvecs, _, _ = cv2.aruco.calibrateCameraCharuco(
            charucoCorners = all_corners,
            charucoIds = all_ids,
            board = board,
            imageSize = im_size,
            cameraMatrix = None,
            distCoeffs = None
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
        ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(ch_corners, ch+ids, board, K, dist, None, None)
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

def extrinsics_from_shared_board(data_root, cams, pattern, rows, cols, square, calibs: Dict[str, cameraCalib], 
                                 charuco_marker = None, dict_name = "DICT_4X4_50"):
    """
    For frames where multiple cameras see the board at (roughly) the same time,
    we estimate each cam's pose wrt the board, then compute relative transforms.
    cam_i -> cam0 by T_i0 = t_i_board * inv(T_0_board)
    """
    from collections import defaultdict
    def se3_log(T):
        # simple log map using cv2.Rodrigues inverse
        R = T[:3, :3]
        t = T[:3, 3:4]  # potential issue here so check later
        rvec, _ = cv2.Rodrigues(R)
        return np.concatenate([rvec.flatten(), t])
    
    def se3_exp(xi):
        rvec = xi[:3].reshape(3, 1)
        t = xi[3:].reshape(3, 1)
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3:4] = t
        return T
    # build per-frame estimates by matching file indices across cams
    # assumes images named like img_0001.jpg across all cams
    frame_to_Tc_board = defaultdict(dict) 
    # collect sorted basenames per cam
    cam_files = {}
    for c in cams:
        files = sorted(glob.glob(os.path.join(data_root, c, "*.jpg")) + 
                       glob.glob(os.path.join(data_root, c, "*.png")))
        cam_files[c] = files
    # Match by index min length
    min_len = min(len(v) for v in cam_files.values())
    for idx in range(min_len):
        for c in cams:
            p = cam_files[c][idx]
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            K = calibs[c].K
            dist = calibs[c].dist
            found, rvec, tvec = estimate_board_pose(img, pattern, rows, cols, square, K, dist,
                                                    charuco_marker=charuco_marker, dict_name=dict_name)
            if not found:
                continue
            T_c_board = rvec_tvec_to_rt(rvec, tvec)
            frame_to_Tc_board[idx][c] = T_c_board
            # Compute relative transforms to cam0 per frame, then aggregate
            ref = cams[0]
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
            # robust average via log/exp
            extrinsics = {}
            for c in cams:
                if c == ref:
                    extrinsics[c] = np.eye(4)
                    continue
                Ts = rel_lists[c]
                if len(Ts) == 0:
                    raise RuntimeError(f"No shared board views found between {ref} and {c}.")
                xis = np.stack([se3_log(T) for T in Ts], axis=0)
                median_xi = np.median(xis, axis=0)
                extrinsics[c] = se3_exp(median_xi)
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