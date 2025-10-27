"""
This program creates a virtual mouse using OpenCV, MediaPipe, and PyAutoGUI.

How it works:
1. The webcam tracks your hand in real time using MediaPipe Hands.
2. The index finger tip controls the mouse cursor position on the screen.
3. When the thumb and index finger tips come close together, a click event is triggered.
4. The mouse movement is mirrored (so it feels natural) and scaled to screen resolution.
5. Press 'q' to quit.

Main idea:
- Move your index finger to move the cursor.
- Tap your thumb and index finger together to click.
"""

import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Get screen size
screen_w, screen_h = pyautogui.size()

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # width
cap.set(4, 480)  # height

click_down = False  # prevents multiple clicks while fingers stay close

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally (mirror effect)
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Index fingertip coordinates
            index_x = int(hand_landmarks.landmark[8].x * w)
            index_y = int(hand_landmarks.landmark[8].y * h)

            # Thumb tip coordinates
            thumb_x = int(hand_landmarks.landmark[4].x * w)
            thumb_y = int(hand_landmarks.landmark[4].y * h)

            # Convert camera coordinates to screen coordinates
            screen_x = np.interp(index_x, (0, w), (0, screen_w))
            screen_y = np.interp(index_y, (0, h), (0, screen_h))
            pyautogui.moveTo(screen_x, screen_y)

            # Distance between index and thumb
            distance = np.hypot(thumb_x - index_x, thumb_y - index_y)

            # Click when fingers are close
            if distance < 40:
                if not click_down:
                    pyautogui.click()
                    click_down = True
            else:
                click_down = False

    cv2.imshow("Virtual Mouse Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
