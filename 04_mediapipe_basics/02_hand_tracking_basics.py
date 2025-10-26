"""
Hand Tracking Basics using MediaPipe

This script detects hands from webcam input and draws landmarks (joints).
It can track up to 2 hands and show finger tips positions.
"""

import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,       # False = live video
    max_num_hands=2,               # Track up to 2 hands
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Drawing utils for landmarks and connections
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for natural selfie view (optional)
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame and detect hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw all 21 landmarks and connections
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

            # Optional: print finger tip coordinates
            # Finger tips are landmarks 4, 8, 12, 16, 20
            h, w, _ = frame.shape
            tips = [4, 8, 12, 16, 20]
            for i in tips:
                x = int(hand_landmarks.landmark[i].x * w)
                y = int(hand_landmarks.landmark[i].y * h)
                cv2.circle(frame, (x, y), 8, (0,0,255), cv2.FILLED)

    # Show result
    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
