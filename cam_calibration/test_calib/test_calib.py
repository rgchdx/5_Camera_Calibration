import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import cv2
import numpy as np
from calibrate_5cams import detect_checkerboard, make_charuco_board

# --- Test 1: Checkerboard detection ---
img = cv2.imread("choriginal.jpg") 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

found, corners = detect_checkerboard(gray, rows=6, cols=9)
print("Checkerboard found:", found)
if found:
    img_vis = cv2.drawChessboardCorners(img, (9, 6), corners, found)
    cv2.imshow("Detected Corners", img_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- Test 2: Charuco board creation ---
aruco_dict, board = make_charuco_board(rows=5, cols=7, square=0.04, marker_len=0.02, dict_name="DICT_4X4_50")
print("ChArUco board created successfully!")