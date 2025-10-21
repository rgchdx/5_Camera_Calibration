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