# YOLO Slicing - Segmentation Scripts

This repository contains three Python scripts utilizing YOLOv11s-seg for object segmentation in videos. These scripts progressively enhance object detection accuracy through slicing techniques, making them effective for detecting small or partially visible objects. 

---

## Scripts Overview

### 1. `yoloDetect.py`
- **Description:** Performs standard YOLO segmentation on an entire video frame. Ideal for detecting objects in scenes with minimal occlusion or overlap.
- **Key Features:**
  - Processes the entire frame in one step.
  - Outputs annotated videos with detected objects and confidence scores.
  - Simple and efficient for general use cases.
- **Usage:** Best suited for quick segmentation tasks with standard video footage.

---

### 2. `yoloSliceDetect.py`
- **Description:** Enhances object detection by dividing the frame into 4 overlapping slices (2 rows × 2 columns). This approach helps in detecting smaller objects.
- **Key Features:**
  - Utilizes 10% overlap between slices for seamless object detection.
  - Processes each slice independently to improve accuracy for small objects.
  - Combines results from all slices into a unified annotated video.
- **Usage:** Recommended for videos with smaller or partially obscured objects.

---

### 3. `yoloSuperSliceDetect.py`
- **Description:** Maximizes detection accuracy by dividing the frame into 12 overlapping slices (4 rows × 3 columns). This method is optimized for complex scenes with very small or occluded objects.
- **Key Features:**
  - Uses 10% overlap between slices for superior coverage.
  - Processes a higher number of slices to ensure no small object is missed.
  - Balances detection quality, speed, and quantity effectively.
- **Usage:** Ideal for videos with high object density or complex backgrounds.

---

## How to Use

1. **Setup:**
   - Install required libraries using:
     ```bash
     pip install ultralytics opencv-python numpy
     ```
   - Ensure the YOLO model file `yolo11s-seg.pt` is in the same directory as the scripts.
   - Place the video file (e.g., `input3.mp4`) in the directory.

2. **Run the Scripts:**
   - For single-frame segmentation:
     ```bash
     python yoloDetect.py
     ```
   - For 4-slice segmentation:
     ```bash
     python yoloSliceDetect.py
     ```
   - For 12-slice segmentation:
     ```bash
     python yoloSuperSliceDetect.py
     ```

3. **Output:**
   - Annotated video files will be saved as `instance-segmentation3.avi`.

---

## Video Explanation

For a detailed explanation and demonstration of these scripts, watch the [YouTube video](https://youtu.be/s6mD_gSRkkI).

---

## Learn More

Visit [Consulting Joe](https://consultingjoe.com) for more AI-powered solutions and tutorials.

---
