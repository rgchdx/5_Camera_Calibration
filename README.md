# Multi-Camera Calibration (5-Camera Setup)

This repository provides a complete pipeline for intrinsic and extrinsic calibration of a multi-camera system (validated on a 5-camera rig) using OpenCV. It is modular, accurate, and compatible with downstream computer vision, XR, and reconstruction workflows.  
This repository is a part of the honors project, “Integrating Artificial Intelligence in a Single-Camera XR-Based Approach for Remote 3D Visualization.”

## Features
- ✅ Supports **Checkerboard** and **ChArUco** calibration targets  
- ✅ Compatible with **Pinhole** and **Fisheye** camera models  
- ✅ Performs **per-camera intrinsic calibration**  
- ✅ Computes **full multi-view extrinsic calibration** relative to a reference camera (`cam0`)  
- ✅ Outputs results in **YAML/JSON** for seamless integration into AR, robotics, and reconstruction pipelines  

## Folder Structure
cam_calibration/

│

├── test_calib/

│   ├── cam_extrinsics_test/

│   │   ├── IMG_4036.png   ← for testing

│   │   ├── IMG_4037.png

│   │   ├── IMG_4038.png

│   │   ├── IMG_4042.png

│   │   ├── IMG_4043.png

│   │   ├── charuco_board_generated.png   ← generated from code

│   │   ├── checkerboard.png   ← generated from code

│   │   ├── choriginal.jpg

│   │   ├── test_calib.py   ← for testing the script)

│   │   └── calibrate_5cams.py   ← main script

│

├── configs/

│   └── sample_calibration_config.yaml  ← Output containing necessary parameters

│

└── README.md

### 1. Pull the repository
``` bash
git clone <repo_url>
cd cam_calibration
```
### 2. Install Python dependencies
``` pip install opencv-python opencv-contrib-python numpy pyyaml ```

## Usage
Run the main calibration script:

``` --data_root ``` Root folder containing images or per-camera subfolders

``` --cams ``` List of camera folder names

``` --pattern ``` Calibration target type: ``` checkerboard ``` or ``` charuco ```

``` --rows ``` Checkerboard inner rows or ChArUco rows (squares)

``` --cols ``` Checkerboard inner cols or ChArUco cols (squares)

``` --square ``` Square size in meters

``` --marker ``` ChArUco marker size in meters (required for ChArUco)

``` --dict ``` ArUco dictionary name (default: ``` DICT_4X4_50 ```

``` --model ``` Camera model: ``` pinhole ``` or ``` fisheye ```

``` --output ``` Output YAML/JSON file path

## Output
The script saves a YAML/JSON file containing:

- Intrinsic matrix (```K```) and distortion coefficients (```dict```) for each camera
- Resolution and camera model
- Extrinsic transformation matrices relative to the reference camera

Example YAML snippet:
```
reference_camera: cam0
cameras:
  cam0:
    model: pinhole
    resolution:
      width: 1920
      height: 1080
    K:
      - [1200.0, 0.0, 960.0]
      - [0.0, 1200.0, 540.0]
      - [0.0, 0.0, 1.0]
    dist: [0.01, -0.02, 0.0, 0.0, 0.0]
    T_cam_ref: [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
```
## Notes
- ChArUco calibration supports pinhole cameras only; fisheye for ChArUco is not supported in this implementation.
- Ensure sufficient images per camera (more than 10 are recommended) for reliable calibration
- For multi-view extrinsics, the script assumes multiple cameras observe the same board at approximately the same time.







