"""
01_opencv_mediapipe_combined.py
Left: Webcam + Face Mesh
Right: Emoji appears only if mouth is open
"""

import cv2
import mediapipe as mp
import numpy as np

# --- Initialize MediaPipe Face Mesh ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# --- Load emoji ---
emoji = cv2.imread('assets/images/yawn.jpg', cv2.IMREAD_UNCHANGED)

# --- Open webcam ---
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # width
cap.set(4, 720)   # height

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # Draw face mesh
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1)
            )

            # --- Mouth open detection ---
            # landmark indices for upper & lower lips
            top_lip = face_landmarks.landmark[13]
            bottom_lip = face_landmarks.landmark[14]

            lip_dist = abs(top_lip.y - bottom_lip.y) * h  # pixel distance

            mouth_open = lip_dist > 15  # threshold, ayarla isteğine göre

            # --- Show emoji if mouth open ---
            if mouth_open:
                emoji_resized = cv2.resize(emoji, (w//2, h))
                if emoji_resized.shape[2] == 4:
                    alpha = emoji_resized[:, :, 3] / 255.0
                    for c in range(3):
                        frame[:, w//2:, c] = alpha*emoji_resized[:, :, c] + (1-alpha)*frame[:, w//2:, c]
                else:
                    frame[:, w//2:] = emoji_resized

    cv2.imshow("Face Mesh + Emoji", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
