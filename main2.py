import cv2
import mediapipe as mp
import numpy as np

# Thresholds
EYE_AR_THRESH = 0.23
MOUTH_AREA_THRESH = 0.12  # Adjust as needed
CONSEC_FRAMES = 15
YAWN_FRAMES = 22  # adjust as needed (~0.5 sec)
yawn_frame_counter = 0

# Landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
INNER_MOUTH = [78, 80, 81, 82, 13, 312, 311, 310, 308, 415, 402, 317]

# Functions
def aspect_ratio(landmarks, indices):
    points = np.array([(landmarks[i][0], landmarks[i][1]) for i in indices])
    A = np.linalg.norm(points[1] - points[5])
    B = np.linalg.norm(points[2] - points[4])
    C = np.linalg.norm(points[0] - points[3])
    return (A + B) / (2.0 * C)

def polygon_area(pts):
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def normalized_mouth_area(landmarks):
    mouth_pts = np.array([landmarks[i] for i in INNER_MOUTH])
    area = polygon_area(mouth_pts)

    left_eye_center = np.mean([landmarks[i] for i in LEFT_EYE], axis=0)
    right_eye_center = np.mean([landmarks[i] for i in RIGHT_EYE], axis=0)
    eye_dist = np.linalg.norm(left_eye_center - right_eye_center)
    if eye_dist == 0:
        return 0
    return area / (eye_dist ** 2)

# Setup MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# State
frame_counter = 0

# Start webcam
cap = cv2.VideoCapture(1)  # Change index if needed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = [(int(l.x * w), int(l.y * h)) for l in face_landmarks.landmark]

            # Eye aspect ratio
            left_ear = aspect_ratio(landmarks, LEFT_EYE)
            right_ear = aspect_ratio(landmarks, RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2.0

            # Mouth area
            mouth_area = normalized_mouth_area(landmarks)

            # Show EAR and mouth area
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, h - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"Mouth Area: {mouth_area:.3f}", (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Drowsiness check
            if avg_ear < EYE_AR_THRESH:
                frame_counter += 1
                if frame_counter >= CONSEC_FRAMES:
                    cv2.putText(frame, "DROWSINESS ALERT", (30, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            else:
                frame_counter = 0

            # Yawning check
            if mouth_area > MOUTH_AREA_THRESH:
                yawn_frame_counter += 1
                if yawn_frame_counter >= YAWN_FRAMES:
                    cv2.putText(frame, "YAWNING", (30, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
            else:
                yawn_frame_counter = 0

            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame, face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
            )

    cv2.imshow("Drowsiness & Yawning Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
