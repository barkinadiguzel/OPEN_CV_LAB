"""
Optical Flow Visualization using Farneback Method

This script computes dense optical flow between consecutive frames of a video.
- Optical flow: Shows apparent motion of pixels between frames
- Hue (color) represents motion direction
- Brightness represents motion magnitude (speed)
- Helps understand movement patterns in video sequences
"""

import cv2
import numpy as np

# Load video
cap = cv2.VideoCapture('assets/videos/sample_video.mp4')

# Take first frame and convert to grayscale
ret, first_frame = cap.read()
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Create mask for drawing optical flow (HSV)
mask = np.zeros_like(first_frame)
mask[..., 1] = 255  # Saturation to max

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 
                                        0.5, 3, 15, 3, 5, 1.2, 0)

    # Compute magnitude and angle
    magnitude, angle = cv2.cartToPolar(flow[...,0], flow[...,1])
    
    # Hue according to flow direction
    mask[..., 0] = angle * 180 / np.pi / 2
    # Value according to flow magnitude
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    
    # Convert HSV to BGR for display
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

    cv2.imshow('Optical Flow', rgb)

    prev_gray = gray

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
