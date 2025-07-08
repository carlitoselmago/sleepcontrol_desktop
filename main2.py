import cv2
import mediapipe as mp
import numpy as np
import pyscreenrec
from screeninfo import get_monitors
import threading
import time
import os

# Thresholds
EYE_AR_THRESH = 0.23
MOUTH_AREA_THRESH = 0.12
CONSEC_FRAMES = 15
YAWN_FRAMES = 22
yawn_frame_counter = 0

YAWNING = False
EYE_FATIGUE = False
SLEEPING = False

# Landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
INNER_MOUTH = [78, 80, 81, 82, 13, 312, 311, 310, 308, 415, 402, 317]

record_folder = 'recordings/'
os.makedirs(record_folder, exist_ok=True)

# --- Screen Recorder Thread ---
class RecorderThread(threading.Thread):
    def __init__(self, recorder, record_folder):
        super().__init__()
        self.recorder = recorder
        self.record_folder = record_folder
        self.recording_event = threading.Event()
        self.stop_event = threading.Event()
        self.is_recording = False
        self.daemon = True

    def run(self):
        while not self.stop_event.is_set():
            if self.recording_event.is_set() and not self.is_recording:
                main_monitor = sorted(get_monitors(), key=lambda m: m.is_primary, reverse=True)[0]
                self.recorder.start_recording(
                    os.path.join(self.record_folder, "screen.mp4"), 30, {
                        "mon": 1,
                        "left": 0,
                        "top": 0,
                        "width": main_monitor.width,
                        "height": main_monitor.height
                    }
                )
                self.is_recording = True
            elif not self.recording_event.is_set() and self.is_recording:
                self.recorder.stop_recording()
                self.is_recording = False
            time.sleep(0.1)

    def start_recording(self):
        self.recording_event.set()

    def stop_recording(self):
        self.recording_event.clear()

    def shutdown(self):
        self.stop_event.set()
        self.recording_event.clear()
        if self.is_recording:
            self.recorder.stop_recording()
            self.is_recording = False

# Video recorder for webcam
class VideoFrameRecorder:
    def __init__(self, path, fps, width, height):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()

# Functions for detection
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
    return 0 if eye_dist == 0 else area / (eye_dist ** 2)

# Setup
recorder_thread = RecorderThread(pyscreenrec.ScreenRecorder(), record_folder)
recorder_thread.start()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(1)
frame_writer = None
frame_counter = 0
FPS = 20

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
            EYE_FATIGUE = False
            if avg_ear < EYE_AR_THRESH:
                frame_counter += 1
                if frame_counter >= CONSEC_FRAMES:
                    EYE_FATIGUE = True
                    cv2.putText(frame, "DROWSINESS ALERT", (30, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            else:
                frame_counter = 0

            # Yawning check
            YAWNING = False
            if mouth_area > MOUTH_AREA_THRESH:
                yawn_frame_counter += 1
                if yawn_frame_counter >= YAWN_FRAMES:
                    YAWNING = True
                    cv2.putText(frame, "YAWNING", (30, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
            else:
                yawn_frame_counter = 0

            if YAWNING or EYE_FATIGUE:
                if not SLEEPING:
                    print("START RECORDING!!!")
                    SLEEPING = True
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    video_path = os.path.join(record_folder, f"webcam_{timestamp}.mp4")
                    frame_writer = VideoFrameRecorder(video_path, FPS, w, h)
                    recorder_thread.start_recording()
                if frame_writer:
                    frame_writer.write(frame)
            else:
                if SLEEPING:
                    print("STOP RECORDING!!!")
                    SLEEPING = False
                    if frame_writer:
                        frame_writer.release()
                        frame_writer = None
                    recorder_thread.stop_recording()

            mp_drawing.draw_landmarks(
                frame, face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
            )

    cv2.imshow("Drowsiness & Yawning Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
recorder_thread.shutdown()
if frame_writer:
    frame_writer.release()

