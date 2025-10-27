"""
This program uses OpenCV and MediaPipe to detect hand gestures via the webcam.
When only the index finger is open (and all other fingers are closed),
a 'nerd' emoji appears on the right side of the screen.

Steps:
1. The webcam captures live video at 640x480 resolution.
2. MediaPipe detects hand landmarks in each frame.
3. The program checks which fingers are open or closed by comparing tip and pip landmarks.
4. If the gesture matches (index finger up, others down), it overlays the emoji.
5. The left side of the window shows the camera, the right side shows the emoji.
"""

import cv2
import mediapipe as mp
import numpy as np

# Load emoji image (with alpha channel if available)
nerd = cv2.imread('assets/emojis/nerd.jpg', cv2.IMREAD_UNCHANGED)

# Initialize MediaPipe Hands detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Open webcam (640x480)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Output window size (camera + emoji space)
out_w = 640 + 320
out_h = 480

# Function to check if a finger is open
def is_finger_open(hand, tip_idx, pip_idx, h):
    """Return True if finger tip is higher (smaller y) than pip joint"""
    return hand.landmark[tip_idx].y * h < hand.landmark[pip_idx].y * h

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    h, w, _ = frame.shape

    # Convert BGR to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(frame_rgb)

    # Create black canvas and place camera frame on the left
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    canvas[:, :w] = frame

    # If a hand is detected
    if hand_results.multi_hand_landmarks:
        hand = hand_results.multi_hand_landmarks[0]

        # Finger state detection
        index_open = is_finger_open(hand, 8, 6, h)
        middle_closed = not is_finger_open(hand, 12, 10, h)
        ring_closed = not is_finger_open(hand, 16, 14, h)
        pinky_closed = not is_finger_open(hand, 20, 18, h)

        # Condition: only index finger open
        if index_open and middle_closed and ring_closed and pinky_closed:
            emoji_resized = cv2.resize(nerd, (320, 480))

            # Apply alpha blending if emoji has transparency
            if emoji_resized.shape[2] == 4:
                alpha = emoji_resized[:, :, 3] / 255.0
                for c in range(3):
                    canvas[:, w:, c] = (alpha * emoji_resized[:, :, c] +
                                        (1 - alpha) * canvas[:, w:, c]).astype(np.uint8)
            else:
                canvas[:, w:] = emoji_resized

    cv2.imshow("Camera + Nerd Emoji", canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
