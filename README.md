# CV_PR3 - Computer Vision and Deep Learning Pipeline

## Project Overview

This project is a practical Computer Vision and Deep Learning project completed using Python, OpenCV, YuNet, and YOLOv8.

The project starts with classical image-processing techniques and then moves to modern deep-learning-based detection. Finally, YuNet face detection and YOLOv8 object detection are combined into a single real-time webcam pipeline.

### Main Topics

- Morphological image processing
- Bitwise image operations
- Grayscale and colour histograms
- Brightness and contrast adjustment
- YuNet face detection
- Real-time face detection using webcam
- Face privacy blur using Gaussian Blur
- YOLOv8 object detection
- Confidence and IoU threshold experiments
- Per-class object detection summary
- Unified YuNet + YOLOv8 pipeline
- Morphological pre-cleaning
- FPS benchmarking
- Final technique comparison

---

## Project Structure

```text
CV_PR3/
│
├── CV_PR3.ipynb
│
├── data/
│   ├── images/
│   │   ├── project_original.png
│   │   ├── project_grayscale.png
│   │   ├── person1.jpg
│   │   ├── person2.jpg
│   │   ├── person3.jpg
│   │   ├── person4.jpg
│   │   ├── yolo_test1.jpg
│   │   ├── yolo_test2.jpg
│   │   └── yolo_test3.jpg
│   │
│   └── models/
│       └── face_detection_yunet_2023mar.onnx
│
└── outputs/
```

> `yolov8n.pt` is downloaded automatically by Ultralytics when the model is loaded for the first time, if it is not already available locally.

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| OpenCV | Image processing and computer vision |
| NumPy | Numerical array operations |
| Matplotlib | Visualization and plots |
| Pandas | Detection summary tables |
| YuNet | Face detection and facial landmarks |
| YOLOv8 | General object detection |
| Jupyter Notebook | Project implementation |

---

## Installation

Install the required Python packages:

```bash
pip install opencv-python==5.0.0.*
pip install opencv-contrib-python==5.0.0.*
pip install ultralytics
pip install matplotlib
pip install numpy
pip install pandas
```

If the OpenCV version available in your environment is different, use a compatible OpenCV version supported by your Python environment.

---

# Tasks Completed

## Task 1 - Morphological Operations

The following operations were implemented:

- Erosion
- Dilation
- Opening
- Closing
- Kernel shape comparison using RECT, ELLIPSE, and CROSS
- Kernel size comparison using 3x3, 5x5, and 9x9 kernels

Morphological opening was also used as a light preprocessing technique for noise removal.

---

## Task 2 - Bitwise Operations and Histograms

The project includes:

- Bitwise AND
- Bitwise OR
- Bitwise XOR
- Bitwise NOT
- Image masking
- Grayscale histogram
- Colour histogram
- Brightness adjustment
- Contrast adjustment

These operations demonstrate how classical computer vision can be used for image analysis and preprocessing.

---

## Task 3 - YuNet Face Detection

YuNet was used for face detection with OpenCV's `FaceDetectorYN` interface.

The project covers:

- Loading the YuNet ONNX model
- Face detection on static images
- Bounding boxes
- Five facial landmarks
- Confidence scores
- Real-time webcam face detection
- Score threshold comparison
- Privacy blur using Gaussian Blur

Example YuNet setup:

```python
import cv2

detector = cv2.FaceDetectorYN_create(
    "data/models/face_detection_yunet_2023mar.onnx",
    "",
    (img_w, img_h),
    0.9
)

detector.setInputSize((img_w, img_h))
_, faces = detector.detect(img)
```

---

## Task 4 - YOLOv8 Object Detection

YOLOv8 Nano was used for general object detection.

The project includes:

- Pretrained YOLOv8 model loading
- Static image detection
- Real-time webcam detection
- Confidence threshold experiments
- IoU/NMS threshold experiments
- Per-class detection counting
- Detection summary bar chart

Example:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model("data/images/yolo_test1.jpg")
result_image = results[0].plot()
```

---

## Task 5 - Integrated Pipeline and Final Comparison

The final pipeline combines:

```text
Webcam Frame
     |
     +--------------------+
     |                    |
     v                    v
   YuNet                YOLOv8
 Face Detection      Object Detection
     |                    |
     +---------+----------+
               |
               v
       Combined Output
```

### Unified Pipeline

- YuNet face boxes are displayed in **green**.
- YOLOv8 object boxes are displayed in **red**.
- YOLOv8 also displays class labels and confidence scores.

### Morphological Pre-cleaning

A light 3x3 morphological opening operation was tested before detection to compare the detection results with and without preprocessing.

### FPS Benchmarking

The project benchmarks:

1. YuNet face detection alone
2. YOLOv8 object detection alone
3. YuNet + YOLOv8 combined pipeline

The same 100-frame webcam sample is used for the three measurements.

---

# Project Screenshots

## Original Image

![Original Project Image](images/project_original.png)

## Grayscale Image

![Grayscale Project Image](images/project_grayscale.png)

## Original vs Grayscale

![Original and Grayscale Comparison](images/original_and_grayscale.png)

> Additional screenshots can be added to the `screenshots/` folder after capturing the YuNet webcam output, YOLOv8 output, unified pipeline, and FPS benchmark chart.

---

# Results and Observations

### Classical Computer Vision

Morphological operations, bitwise operations, and histograms are computationally lightweight and are useful for preprocessing, masking, noise removal, and image analysis.

### YuNet

YuNet provides lightweight face detection together with five facial landmarks. It is suitable for real-time face detection applications.

### YOLOv8

YOLOv8 provides general object detection for multiple COCO classes such as person, car, dog, chair, and other supported objects.

### Unified Pipeline

Combining YuNet and YOLOv8 allows the system to detect faces and general objects in the same webcam frame. The combined pipeline requires more computation because both deep-learning models process every frame.

---

# Final Comparison

| Technique | Type | Main Purpose | Typical Use |
|---|---|---|---|
| Morphology | Classical | Noise and shape processing | Image preprocessing |
| Bitwise Operations | Classical | Mask and region operations | Image masking |
| Histograms | Classical | Intensity and colour analysis | Brightness/contrast analysis |
| YuNet | Deep Learning | Face detection | Real-time face detection |
| YOLOv8 | Deep Learning | General object detection | Object detection |
| YuNet + YOLOv8 | Deep Learning Pipeline | Face + object detection | Real-time monitoring |

---

# Conclusion

This project demonstrates a complete progression from classical image processing to deep-learning-based computer vision.

Classical techniques are useful for fast preprocessing and image analysis. YuNet provides lightweight face detection, while YOLOv8 provides general-purpose object detection. Combining both models creates an integrated real-time computer vision pipeline.

The final deployment choice should consider detection quality, FPS, hardware capability, and application requirements. For a low-power edge device, lightweight models such as YuNet and YOLOv8n are suitable starting points. A cloud server can support experiments with larger models or higher input resolutions.

---

# How to Run

1. Open the `CV_PR3` folder in VS Code or Jupyter Notebook.
2. Install the required packages.
3. Make sure the YuNet ONNX model is present in `data/models/`.
4. Make sure the test images are present in `data/images/`.
5. Open `CV_PR3.ipynb`.
6. Run the notebook cells from Task 1 through Task 5 in order.
7. For webcam tasks, allow camera access when requested.
8. Press `Q` in the OpenCV webcam window to stop real-time detection.

---

# Project Information

**Project:** PR-3 Computer Vision and Deep Learning  
**Notebook:** `CV_PR3.ipynb`  
**Main Models:** YuNet and YOLOv8n  
**Frameworks:** OpenCV and Ultralytics  
**Environment:** Python / Jupyter Notebook

---

# License and Educational Use

This repository is intended for academic and educational purposes. The project uses OpenCV and Ultralytics pretrained models and follows the applicable licenses of those components.
