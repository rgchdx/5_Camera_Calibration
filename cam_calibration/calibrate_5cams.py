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

