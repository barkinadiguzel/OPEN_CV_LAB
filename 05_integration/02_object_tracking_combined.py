"""
02_object_tracking_combined.py
Track an object selected by the user on webcam feed.
Left: Live webcam
Right: Bounding box showing tracked object
"""

import cv2

# --- Open webcam ---
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# --- Read first frame ---
ret, frame = cap.read()
if not ret:
    print("Camera not detected.")
    exit()

# --- Let user select ROI (Region of Interest) ---
roi = cv2.selectROI("Select Object", frame, False)
x, y, w, h = roi
tracker = cv2.TrackerCSRT_create()  # You can choose KCF, MIL, TLD, MOSSE, CSRT
tracker.init(frame, roi)
cv2.destroyWindow("Select Object")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Update tracker
    success, roi = tracker.update(frame)
    if success:
        x, y, w, h = tuple(map(int, roi))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(frame, "Tracking", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Lost", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Object Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
