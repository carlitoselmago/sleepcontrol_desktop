import cv2
import time
import os
import numpy as np
from datetime import datetime
from threading import Thread, Lock
from collections import deque
from multiprocessing import Process, Queue, Pipe
import platform
import subprocess
import sys

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    import psutil
except ImportError:
    psutil = None


# ==============================
# CONFIG
# ==============================
SHOW_WINDOW = True


# ==============================
# Utils
# ==============================
def resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


# ==============================
# Camera Reader Thread
# ==============================
class CameraReader(Thread):
    def __init__(self, camera_id=0, resolution="640x480"):
        super().__init__()
        self.camera_id = camera_id
        self.resolution = resolution
        self.cap = None
        self.running = True
        self.frame = None
        self.lock = Lock()
        self._init_camera()

    def _init_camera(self):
        backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_V4L2
        self.cap = cv2.VideoCapture(self.camera_id, backend)

        if not self.cap.isOpened():
            print("❌ Failed to open camera")
            self.running = False
            return

        w, h = self.resolution.split("x")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(w))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(h))
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        print("✅ Camera opened")

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.01)

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            frame = self.frame.copy()

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            frame, ts, (10, 30),
            cv2.FONT_HERSHEY_PLAIN, 1.5,
            (255, 255, 255), 1
        )
        return frame

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()


# ==============================
# FFmpeg Writer Process
# ==============================
class FFmpegWriterProcess(Process):
    def __init__(self, frame_queue, resolution, fps, output_path, control_pipe, min_duration):
        super().__init__()
        self.frame_queue = frame_queue
        self.resolution = resolution
        self.fps = fps
        self.output_path = output_path
        self.control_pipe = control_pipe
        self.min_duration = min_duration

    def run(self):
        w, h = map(int, self.resolution.split("x"))
        filename = os.path.join(
            self.output_path,
            f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        )

        frames = []
        timestamps = []
        last_activity = time.time()

        while True:
            if self.control_pipe.poll():
                msg = self.control_pipe.recv()
                if msg == "extend":
                    last_activity = time.time()
                elif msg == "stop":
                    break

            if time.time() - last_activity > self.min_duration:
                break

            try:
                frame = self.frame_queue.get(timeout=0.5)
                frames.append(frame)
                timestamps.append(time.time())
            except Exception:
                pass

        fps = (
            len(timestamps) / max(0.01, timestamps[-1] - timestamps[0])
            if len(timestamps) >= 2
            else self.fps
        )

        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}",
            "-r", str(fps),
            "-i", "-",
            "-an",
            "-vcodec", "libx264",
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            filename,
        ]

        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for f in frames:
            proc.stdin.write(f.tobytes())
        proc.stdin.close()
        proc.wait()


# ==============================
# Sleep Control (MediaPipe Tasks)
# ==============================
class SleepControl:
    def __init__(self, **kwargs):
        self.on_sleep = []
        self.on_wake = []

        self.interval = kwargs.get("interval", 0.15)
        self.camera_id = kwargs.get("camera_id", 0)
        self.threshold = kwargs.get("threshold", 0.22)
        self.sleepsum = kwargs.get("sleepsum", 3)

        self.output_dir = resource_path(kwargs.get("output_dir", "videos"))
        self.resolution = kwargs.get("resolution", "640x480")
        self.min_recording_duration = kwargs.get("min_recording_duration", 7)

        os.makedirs(self.output_dir, exist_ok=True)

        self.judges = deque(maxlen=self.sleepsum)
        self.recording_active = False
        self.was_sleeping = False

        self.frame_queue = None
        self.recorder_pipe = None
        self.current_recorder = None

        self.fps = 25
        self.recording_interval = 1 / self.fps
        self.running = False

        self.camera = CameraReader(self.camera_id, self.resolution)

        # -------- MediaPipe Tasks --------
        model_path = resource_path("data/face_landmarker.task")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "Missing model file: data/face_landmarker.task"
            )

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)

        self.last_landmarks = None

        # MediaPipe EAR landmark indices
        self.LEFT_EYE = {
            "upper": [159, 158],
            "lower": [145, 144],
            "corners": [33, 133],
        }
        self.RIGHT_EYE = {
            "upper": [386, 385],
            "lower": [374, 373],
            "corners": [362, 263],
        }

    # ---------------------------------
    # EAR
    # ---------------------------------
    def _eye_aspect_ratio(self, landmarks, eye):
        def dist(a, b):
            return np.linalg.norm(landmarks[a] - landmarks[b])

        A = dist(eye["upper"][0], eye["lower"][0])
        B = dist(eye["upper"][1], eye["lower"][1])
        C = dist(eye["corners"][0], eye["corners"][1])

        if C == 0:
            return 0.0
        return (A + B) / (2.0 * C)

    # ---------------------------------
    # Sleep detection
    # ---------------------------------
    def detect_sleep(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = self.face_landmarker.detect(mp_image)

        if not result.face_landmarks:
            self.last_landmarks = None
            self._handle_wake()
            return False

        h, w, _ = frame.shape
        landmarks = np.array([
            [int(p.x * w), int(p.y * h)]
            for p in result.face_landmarks[0]
        ])

        self.last_landmarks = landmarks

        left_ear = self._eye_aspect_ratio(landmarks, self.LEFT_EYE)
        right_ear = self._eye_aspect_ratio(landmarks, self.RIGHT_EYE)
        ear = (left_ear + right_ear) / 2.0

        self.judges.append(ear)
        avg = np.mean(self.judges)
        print("👁️ EAR:", round(avg, 3))

        if avg < self.threshold:
            if not self.recording_active:
                self._start_recording()
            else:
                self.recorder_pipe.send("extend")
            return True
        else:
            self._handle_wake()
            return False

    # ---------------------------------
    # Recording control
    # ---------------------------------
    def _start_recording(self):
        self.recording_active = True
        self.frame_queue = Queue(maxsize=100)
        parent, child = Pipe()
        self.recorder_pipe = parent

        w, h = self.resolution.split("x")
        self.current_recorder = FFmpegWriterProcess(
            self.frame_queue,
            f"{w}x{h}",
            self.fps,
            self.output_dir,
            child,
            self.min_recording_duration,
        )
        self.current_recorder.start()

        self.was_sleeping = True
        for cb in self.on_sleep:
            cb()

    def _handle_wake(self):
        if self.was_sleeping:
            for cb in self.on_wake:
                cb()
            self.was_sleeping = False

    # ---------------------------------
    # Main loop
    # ---------------------------------
    def start_capturing(self):
        self.camera.start()
        self.running = True
        last_detect = time.time()
        last_frame = time.time()

        try:
            while self.running:
                frame = self.camera.get_frame()
                if frame is None:
                    time.sleep(0.005)
                    continue

                now = time.time()
                if now - last_detect >= self.interval:
                    self.detect_sleep(frame)
                    last_detect = now

                if SHOW_WINDOW and self.last_landmarks is not None:
                    for (x, y) in self.last_landmarks:
                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

                if self.recording_active and self.frame_queue and not self.frame_queue.full():
                    if now - last_frame >= self.recording_interval:
                        self.frame_queue.put(frame)
                        last_frame = now

                if SHOW_WINDOW:
                    cv2.imshow("SleepControl (MediaPipe Tasks)", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        finally:
            self.stop_capturing()

    def stop_capturing(self):
        self.running = False
        if self.current_recorder:
            self.recorder_pipe.send("stop")
            self.current_recorder.join()
        self.camera.stop()
        if SHOW_WINDOW:
            cv2.destroyAllWindows()
