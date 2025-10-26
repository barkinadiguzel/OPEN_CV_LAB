"""
Facial Expression - Mouth Openness Detection

This script detects a face using MediaPipe and measures the mouth opening.
- Useful for detecting smile, talking, or just tracking mouth movements.
"""

import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

# Function to compute Euclidean distance between two points
def euclidean_dist(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw the face mesh
            mp_drawing.draw_landmarks(
                frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1)
            )

            # Landmarks for mouth top and bottom
            top_lip = face_landmarks.landmark[13]   # Upper lip center
            bottom_lip = face_landmarks.landmark[14] # Lower lip center

            # Convert normalized coordinates to pixels
            top = (int(top_lip.x * w), int(top_lip.y * h))
            bottom = (int(bottom_lip.x * w), int(bottom_lip.y * h))

            # Draw circles on lips
            cv2.circle(frame, top, 5, (0,0,255), -1)
            cv2.circle(frame, bottom, 5, (0,0,255), -1)

            # Calculate mouth opening
            mouth_open = euclidean_dist(top, bottom)
            cv2.putText(frame, f"Mouth Open: {mouth_open:.1f}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

    cv2.imshow("Mouth Openness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
