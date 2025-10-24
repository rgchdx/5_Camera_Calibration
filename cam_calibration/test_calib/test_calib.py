import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import cv2
import numpy as np
import shutil
from calibrate_5cams import detect_checkerboard, make_charuco_board, detect_charuco, objpoints_checkerboard, intrinsic_calibration, estimate_board_pose, extrinsics_from_shared_board

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

# --- Test 3: Detect ChArUco corners on real image ---
IMG_PATH = "IMG_4043.png"

aruco_dict, board = make_charuco_board(
    rows=5, cols=7,
    square=0.04,         # 4 cm square size
    marker_len=0.02,     # 2 cm marker size
    dict_name="DICT_4X4_50"
)
print("[INFO] ChArUco board object created.")

img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError(f"Could not read {IMG_PATH}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
found, ch_corners, ch_ids = detect_charuco(gray, aruco_dict, board)
print(f"[INFO] ChArUco detected: {found}")

if found:
    img_vis = img.copy()
    cv2.aruco.drawDetectedCornersCharuco(img_vis, ch_corners, ch_ids, (0, 255, 0))
    print(f"[INFO] Detected {len(ch_ids)} ChArUco corners.")
    disp = cv2.resize(img_vis, (1280, int(img_vis.shape[0] * 1280 / img_vis.shape[1])))
    cv2.imshow("Detected ChArUco Corners", disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("[WARN] No ChArUco corners detected. Check lighting, focus, or board visibility.")
    exit()

# --- Test 4: Intrinsic calibration (single or multiple ChArUCo images) ---
print("\n[INFO] Running intrinsic calibration...")
rms, calib = intrinsic_calibration(
    image_paths=[IMG_PATH],  # ideally multiple images for accuracy
    pattern="charuco",
    rows=5, cols=7,
    square=0.04,
    charuco_marker=0.02,
    model="pinhole"
)

print(f"[RESULT] RMS error: {rms:.6f}")
print("Camera matrix K:\n", calib.K)
print("Distortion coefficients:\n", calib.dist.ravel())

# --- Test 5: Pose estimation and visualization --- 
print("\n[INFO] Estimating ChArUco board pose...")
found, rvec, tvec = estimate_board_pose(
    gray=gray,
    pattern="charuco",
    rows=5, cols=7,
    square=0.04,
    K=calib.K,
    dist=calib.dist,
    charuco_marker=0.02,
    dict_name="DICT_4X4_50"
)

print(f"[RESULT] Pose found: {found}")
if found:
    print("Rotation vector (rvec):\n", rvec)
    print("Translation vector (tvec):\n", tvec)

    # Draw coordinate axes on the original image
    img_vis = img.copy()
    cv2.drawFrameAxes(img_vis, calib.K, calib.dist, rvec, tvec, length=0.05)  # 5 cm axes

    # Resize for display
    max_width = 1280
    scale = max_width / img_vis.shape[1]
    img_disp = cv2.resize(img_vis, (0, 0), fx=scale, fy=scale)
    cv2.imshow("ChArUco Pose Visualization", img_disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("[WARN] Could not estimate pose — ensure enough corners were detected.")

# --- Test 6: Getting extrinsics for multiple cameras ---
