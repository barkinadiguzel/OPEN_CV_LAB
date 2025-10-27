"""
This program uses MediaPipe Pose to detect human body positions in real time.

How it works:
1. Captures frames from the webcam and runs pose estimation on each frame.
2. Extracts body landmarks (shoulder, hip, knee, ankle) from the detected person.
3. Calculates joint angles (knee and hip) using simple trigonometry.
4. Based on these angles:
   - If knee and hip are bent → it detects "Squat"
   - If both are straight → it detects "Plank"
5. Displays the detected move on the video stream.
6. Press 'q' to quit.

Main idea:
- It tracks your left body side and classifies your movement in real time.
- This is a base for future systems like exercise counters or form correction tools.
"""

import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to calculate angle between 3 points (a-b-c)
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB (MediaPipe requirement)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)

    # Convert back to BGR for OpenCV display
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    try:
        landmarks = results.pose_landmarks.landmark

        # Get left side body points
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * 640,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * 480]
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * 640,
               landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * 480]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * 640,
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * 480]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * 640,
                 landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * 480]

        # Calculate knee and hip angles
        knee_angle = calculate_angle(hip, knee, ankle)
        hip_angle = calculate_angle(shoulder, hip, knee)

        # Detect pose based on angle thresholds
        pose_name = "None"
        if knee_angle < 100 and hip_angle < 140:
            pose_name = "Squat"
        elif knee_angle > 160 and hip_angle > 160:
            pose_name = "Plank"

        # Display current pose on screen
        cv2.putText(image, f'Move: {pose_name}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    except:
        pass

    # Draw skeleton landmarks
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
    )

    cv2.imshow('Fitness Pose Estimator', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
