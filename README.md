# 🎥 Vision Lab

A hands-on laboratory for learning, experimenting, and mastering the fundamentals of Computer Vision using OpenCV, MediaPipe, and Python.
This repository takes you on a complete journey — from the pixel-level basics of how an image works, to advanced real-time applications such as gesture control, pose tracking, and motion estimation.

Whether you’re just getting started or refining your skills, each section builds upon the previous one — guiding you through image processing, feature detection, and intelligent vision systems step by step.
The goal isn’t just to use vision algorithms, but to understand how machines actually see the world.

> 💡 Perfect for students, researchers, and curious minds who love experimenting with visual intelligence.
---
# 🧩 Note on Advanced Topics

- The last modules in the 03_advanced/ section —
07_camera_calibration.py, 08_pose_estimation.py, and 09_depth_estimation.py —
touch on concepts that are more mathematical and theoretical (camera matrices, 3D geometry, projection models, etc.).

If these feel overwhelming at first, don’t get stuck — it’s completely normal.
You can treat them as bonus explorations for later. Once you’ve grasped image filtering, edge detection, and tracking, these advanced concepts will make a lot more sense.
The idea is to expose you to the depth of computer vision, not to master it all in one go.
---

## 🧠 Learning Goals

- Understand image representation and manipulation.

- Learn key OpenCV techniques (filters, transformations, thresholding).

- Explore advanced CV algorithms (edges, features, flow).

- Use MediaPipe for real-time face, hand, and pose tracking.

- Build fun, interactive projects (Virtual Mouse, Fitness Estimator, etc).

---

## 🧩 Project Structure
```
vision-lab/
│
├── README.md
├── requirements.txt
│
├── 01_basics/
│   ├── 01_what_are_images.py
│   ├── 02_read_and_write_images.py
│   ├── 03_read_and_write_videos.py
│   ├── 04_pixels_and_channels.py
│   ├── 05_color_spaces.py
│   └── 06_image_resizing.py
│
├── 02_image_processing/
│   ├── 01_image_histogram.py
│   ├── 02_2d_convolution.py
│   ├── 03_average_filtering.py
│   ├── 04_median_filtering.py
│   ├── 05_gaussian_filtering.py
│   ├── 06_thresholding.py
│   └── 07_segmentation_basics.py
│
├── 03_advanced/
│   ├── 01_gradients_and_edges.py
│   ├── 02_canny_edge_detection.py
│   ├── 03_hough_line_transform.py
│   ├── 04_harris_corner_detection.py
│   ├── 05_sift_feature_detection.py
│   ├── 06_optical_flow.py
│   ├── 07_camera_calibration.py
│   ├── 08_pose_estimation.py
│   └── 09_depth_estimation.py
│
├── 04_mediapipe_basics/
│   ├── 01_face_mesh_basics.py
│   ├── 02_hand_tracking_basics.py
│   ├── 03_pose_tracking_basics.py
│   └── 04_facial_expression_basics.py
│
├── 05_integration/
│   ├── 01_opencv_mediapipe_combined.py
│   ├── 02_object_tracking_combined.py
│   └── 03_hands_face_pose_fusion.py
│
├── 06_projects/
│   ├── 01_real_time_emoji_display.py
│   ├── 02_virtual_mouse_control.py
│   └── 03_fitness_pose_estimator.py
│   
│
└── assets/
    ├── images/ and /calibration in here
    ├── videos/
    └── emojis/
    

```
---

## 🚀 Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/vision-lab.git
cd vision-lab

# Create a virtual environment
conda create -n visionlab python=3.10
conda activate visionlab

# Install dependencies
pip install -r requirements.txt
```
---
## 📬Feedback
For feedback or questions, contact: [barkin.adiguzel@gmail.com](mailto:barkin.adiguzel@gmail.com)






