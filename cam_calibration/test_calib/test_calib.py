import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import cv2
import numpy as np
from calibrate_5cams import detect_checkerboard, make_charuco_board, detect_charuco

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
#cv2.imshow("ChArUco Board", img_board)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#cv2.imwrite("charuco_board_generated.png", img_board)
#print("ChArUco board image saved as charuco_board_generated.png")

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

