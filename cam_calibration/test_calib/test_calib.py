import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import cv2
import numpy as np
from calibrate_5cams import detect_checkerboard, make_charuco_board, detect_charuco, objpoints_checkerboard, intrinsic_calibration, estimate_board_pose

# --- Test 1: Checkerboard detection ---
img = cv2.imread("checkerboard.png") 
print("Image loaded:", img is not None)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

found, corners = detect_checkerboard(gray, rows=9, cols=6)
print("Checkerboard found:", found)
#if found:
#    img_vis = cv2.drawChessboardCorners(img, (9, 6), corners, found)
#    cv2.imshow("Detected Corners", img_vis)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()

# --- Test 2: Charuco board creation ---
aruco_dict, board = make_charuco_board(rows=5, cols=7, square=0.04, marker_len=0.02, dict_name="DICT_4X4_50")
print("ChArUco board created successfully!")
board_size = (700, 500)
img_board = cv2.aruco.drawPlanarBoard(board, board_size, marginSize=10, borderBits=1)
cv2.imshow("ChArUco Board", img_board)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("charuco_board_generated.png", img_board)
print("ChArUco board image saved as charuco_board_generated.png")

# --- Test 3: Charuco detection ---
img = cv2.imread("charuco_board_generated.png")
print("Image loaded:", img is not None)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
found, ch_corners, ch_ids = detect_charuco(gray, aruco_dict, board)
print("ChArUco detected:", found)
if found:
    img_vis = img.copy()
    cv2.aruco.drawDetectedCornersCharuco(img_vis, ch_corners, ch_ids, (0, 255, 0))
    cv2.imshow("Detected ChArUco Corners", img_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- Test 4: Object points generation for checkerboard ---
objp = objpoints_checkerboard(rows=9, cols=6, square=0.025)
print("Object points for checkerboard:\n", objp)
print("Number of object points:", objp.shape[0])

# --- Test 5: Intrinsic calibration using ChArUco images ---
rms, calib = intrinsic_calibration(
    image_paths=["IMG_4034.png"],   # replace with multiple images if available
    pattern="charuco",
    rows=5,
    cols=7,
    square=0.04,          # in meters
    charuco_marker=0.02,  # in meters
    model="pinhole"
)
print("RMS error:", rms)
print("Camera matrix:\n", calib.K)
print("Distortion coefficients:\n", calib.dist)

# --- Test 6: Pose estimation using ChArUco board ---
img = cv2.imread("IMG_4034.png")
print("Original image dimensions:", img.shape)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
found, rvec, tvec = estimate_board_pose(
    gray=gray,
    pattern="charuco",
    rows=5,
    cols=7,
    square=0.04,
    K=calib.K,
    dist=calib.dist,
    charuco_marker=0.02,
    dict_name="DICT_4X4_50"
)
print("ChArUco board pose found:", found)
if found:
    print("Rotation vector (rvec):\n", rvec)
    print("Translation vector (tvec):\n", tvec)

    # Draw axes on a copy of the real image
    img_vis = img.copy()
    cv2.drawFrameAxes(img_vis, calib.K, calib.dist, rvec, tvec, length=0.1)

    # Resize for display to fit on screen
    max_width = 1200
    scale = max_width / img_vis.shape[1]
    img_disp = cv2.resize(img_vis, (0, 0), fx=scale, fy=scale)

    cv2.imshow("ChArUco Pose", img_disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()