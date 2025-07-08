import cv2
import mediapipe as mp
import numpy as np
import pyscreenrec
from screeninfo import get_monitors
import multiprocessing
import time
import os
import subprocess

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

def screen_record_worker(record_folder, start_event, stop_event):
    recorder = pyscreenrec.ScreenRecorder()
    is_recording = False
    while not stop_event.is_set():
        if start_event.is_set() and not is_recording:
            main_monitor = sorted(get_monitors(), key=lambda m: m.is_primary, reverse=True)[0]
            recorder.start_recording(
                os.path.join(record_folder, "screen.mp4"), 30, {
                    "mon": 1,
                    "left": 0,
                    "top": 0,
                    "width": main_monitor.width,
                    "height": main_monitor.height
                }
            )
            is_recording = True
        elif not start_event.is_set() and is_recording:
            recorder.stop_recording()
            is_recording = False
        time.sleep(0.1)
    if is_recording:
        recorder.stop_recording()

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

def get_video_duration(path):
    """Return duration in seconds of a video file using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries",
                "format=duration", "-of",
                "default=noprint_wrappers=1:nokey=1", path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return float(result.stdout.strip())
    except Exception:
        return None

def get_video_fps(path):
    """Return FPS of a video using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "default=nokey=1:noprint_wrappers=1",
                path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        rate = result.stdout.strip()
        if "/" in rate:
            num, denom = map(int, rate.split('/'))
            return num / denom
        return float(rate)
    except Exception as e:
        print(f"Failed to get FPS: {e}")
        return 30.0  # fallback

def get_video_duration(path):
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries",
                "format=duration", "-of",
                "default=noprint_wrappers=1:nokey=1", path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return float(result.stdout.strip())
    except Exception:
        return None

def stretch_screen_video(screen_path, webcam_path):
    """Stretch screen video to match webcam duration using correct PTS multiplier."""
    time.sleep(10)

    webcam_duration = get_video_duration(webcam_path)
    screen_duration = get_video_duration(screen_path)

    if webcam_duration is None or screen_duration is None or screen_duration == 0:
        print("⚠️ Could not get valid durations.")
        return

    stretch_ratio = webcam_duration / screen_duration
    print(f"🎬 Stretching screen video: setpts={stretch_ratio:.6f} * PTS")

    temp_output = screen_path + "_temp.mp4"

    cmd = [
        "ffmpeg", "-y", "-i", screen_path,
        "-filter_complex", f"setpts={stretch_ratio:.6f}*PTS",
        "-an", temp_output
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    os.replace(temp_output, screen_path)
    print("✅ Screen video stretched to match webcam duration.")

def stretch_screen_video_process(screen_path, webcam_path):
    p = multiprocessing.Process(
        target=stretch_screen_video,
        args=(screen_path, webcam_path)
    )
    p.start()


def check_camera_access(i):
    cap = cv2.VideoCapture(i)
    time.sleep(1)
    ok = cap.isOpened()
    cap.release()
    return ok

def get_working_camera_index():
   
    for i in range(4):
       
        if check_camera_access(i):
            print(f"✅ Found working camera at index {i}")
            return i
    print("❌ No working camera found.")
    sys.exit(1)

# Setup
record_folder = 'recordings/'
os.makedirs(record_folder, exist_ok=True)

screen_start_event = multiprocessing.Event()
screen_stop_event = multiprocessing.Event()
screen_proc = multiprocessing.Process(
    target=screen_record_worker,
    args=(record_folder, screen_start_event, screen_stop_event)
)
screen_proc.start()

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
cap = cv2.VideoCapture(get_working_camera_index())
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
                    webcam_path = os.path.join(record_folder, f"webcam_{timestamp}.mp4")
                    screen_path = os.path.join(record_folder, f"screen_{timestamp}.mp4")
                    frame_writer = VideoFrameRecorder(webcam_path, FPS, w, h)
                    # Set screen recording output file for this session
                    screen_recording_file = screen_path
                    screen_start_event.set()
                if frame_writer:
                    frame_writer.write(frame)
            else:
                if SLEEPING:
                    print("STOP RECORDING!!!")
                    SLEEPING = False
                    if frame_writer:
                        frame_writer.release()
                        frame_writer = None
                    screen_start_event.clear()
                    # Rename the screen recording to match the timestamp
                    # (Assumes pyscreenrec always writes to 'screen.mp4')
                    screen_tmp = os.path.join(record_folder, "screen.mp4")
                    if os.path.exists(screen_tmp):
                        os.rename(screen_tmp, screen_recording_file)
                        # Launch stretching process
                        stretch_screen_video_process(screen_recording_file, webcam_path)

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

