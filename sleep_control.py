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

import onnxruntime as ort

try:
    import psutil
except ImportError:
    psutil = None


# IMPORTANT:
# When using Flet (or any GUI), OpenCV windows must be disabled on macOS
SHOW_WINDOW = False


# ---------------------------------
# Utils
# ---------------------------------
def resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


# ---------------------------------
# Camera Reader Thread
# ---------------------------------
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
        backend = (
            cv2.CAP_AVFOUNDATION
            if platform.system() == "Darwin"
            else cv2.CAP_V4L2
        )

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
                time.sleep(0.05)

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


# ---------------------------------
# FFmpeg Writer Process
# ---------------------------------
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

        if len(timestamps) >= 2:
            fps = len(timestamps) / max(0.01, timestamps[-1] - timestamps[0])
        else:
            fps = self.fps

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


# ---------------------------------
# Sleep Control (InsightFace 2d106det)
# ---------------------------------
class SleepControl:
    def __init__(self, **kwargs):
        self.on_sleep = []
        self.on_wake = []

        self.interval = kwargs.get("interval", 0.4)
        self.camera_id = kwargs.get("camera_id", 0)
        self.threshold = kwargs.get("threshold", 0.20)  # tuned for 106 landmarks
        self.sleepsum = kwargs.get("sleepsum", 5)
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

        # Face detector (only for bounding box)
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # ONNX landmark model (InsightFace)
        self.landmark_sess = ort.InferenceSession(
            resource_path("data/2d106det.onnx"),
            providers=["CPUExecutionProvider"]
        )
        self.landmark_input = self.landmark_sess.get_inputs()[0].name
        self.landmark_output = self.landmark_sess.get_outputs()[0].name

        self.last_landmarks = None

        # Eye indices for InsightFace 106
        self.LEFT_EYE = [35, 36, 37, 38, 39, 40]
        self.RIGHT_EYE = [89, 90, 91, 92, 93, 94]

    # ---------------------------------
    # Landmark inference
    # ---------------------------------
    def _predict_landmarks(self, frame, face):
        x, y, w, h = face
        face_img = frame[y:y+h, x:x+w]

        if face_img.size == 0:
            return None

        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = cv2.resize(face_img, (192, 192))
        face_img = face_img.astype(np.float32) / 255.0

        face_img = np.transpose(face_img, (2, 0, 1))  # CHW
        face_img = face_img[np.newaxis, :, :, :]      # NCHW

        preds = self.landmark_sess.run(
            [self.landmark_output],
            {self.landmark_input: face_img}
        )[0]

        landmarks = preds.reshape(-1, 2)

        landmarks[:, 0] = landmarks[:, 0] * w + x
        landmarks[:, 1] = landmarks[:, 1] * h + y

        return landmarks.astype(int)

    # ---------------------------------
    # EAR
    # ---------------------------------
    def _eye_aspect_ratio(self, landmarks, idxs):
        p = landmarks[idxs]
        A = np.linalg.norm(p[1] - p[5])
        B = np.linalg.norm(p[2] - p[4])
        C = np.linalg.norm(p[0] - p[3])
        return (A + B) / (2.0 * C)

    # ---------------------------------
    # Sleep detection
    # ---------------------------------
    def detect_sleep(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_detector.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
        )

        if len(faces) == 0:
            self.last_landmarks = None
            self._handle_wake()
            return False

        face = faces[0]
        landmarks = self._predict_landmarks(frame, face)

        if landmarks is None or len(landmarks) < 106:
            self._handle_wake()
            return False

        self.last_landmarks = landmarks

        left_ear = self._eye_aspect_ratio(landmarks, self.LEFT_EYE)
        right_ear = self._eye_aspect_ratio(landmarks, self.RIGHT_EYE)
        ear = (left_ear + right_ear) / 2.0

        self.judges.append(ear)
        avg = np.mean(self.judges)
        print("👁️ EAR:", avg)

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
                    time.sleep(0.01)
                    continue

                now = time.time()
                if now - last_detect >= self.interval:
                    self.detect_sleep(frame)
                    last_detect = now

                if self.recording_active and self.frame_queue and not self.frame_queue.full():
                    if now - last_frame >= self.recording_interval:
                        self.frame_queue.put(frame)
                        last_frame = now

        finally:
            self.stop_capturing()

    def stop_capturing(self):
        self.running = False
        if self.current_recorder:
            self.recorder_pipe.send("stop")
            self.current_recorder.join()
        self.camera.stop()
