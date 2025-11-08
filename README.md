# Multi-Camera Calibration (5-Camera Setup)

This repository provides a complete pipeline for intrinsic and extrinsic calibration of a multi-camera system (validated on a 5-camera rig) using OpenCV. It is modular, accurate, and compatible with downstream computer vision, XR, and reconstruction workflows.  
This repository is a part of the honors project, “Integrating Artificial Intelligence in a Single-Camera XR-Based Approach for Remote 3D Visualization.”

## Features
- ✅ Supports **Checkerboard** and **ChArUco** calibration targets  
- ✅ Compatible with **Pinhole** and **Fisheye** camera models  
- ✅ Performs **per-camera intrinsic calibration**  
- ✅ Computes **full multi-view extrinsic calibration** relative to a reference camera (`cam0`)  
- ✅ Outputs results in **YAML/JSON** for seamless integration into AR, robotics, and reconstruction pipelines  


