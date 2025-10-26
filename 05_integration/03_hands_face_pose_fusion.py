import cv2
import mediapipe as mp

# --- Initialize MediaPipe modules ---
mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)

face_mesh = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True)
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- Open camera ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- Process modules ---
    face_res = face_mesh.process(frame_rgb)
    hands_res = hands.process(frame_rgb)
    pose_res = pose.process(frame_rgb)

    # --- Draw Face Mesh ---
    if face_res.multi_face_landmarks:
        for face_landmarks in face_res.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame, face_landmarks, mp_face.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

    # --- Draw Hands ---
    if hands_res.multi_hand_landmarks:
        for hand_landmarks in hands_res.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

    # --- Draw Pose ---
    if pose_res.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec
        )

    cv2.imshow("MediaPipe Fusion", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
