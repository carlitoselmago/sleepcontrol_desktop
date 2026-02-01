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

from mediapipe.python.solutions.face_mesh import FaceMesh

try:
    import psutil
except ImportError:
    psutil = None

SHOW_WINDOW = True


# ---------------------------------
# Utils
# ---------------------------------
def resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


# ---------------------------------
# Camera Reader
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
        self.initialize_camera()

    def initialize_camera(self):
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

        W, H = self.resolution.split("x")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(W))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(H))
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
                cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 1
            )
            return frame

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()


# ---------------------------------
# FFmpeg Writer Process (unchanged)
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
        W, H = self.resolution.split("x")
        width, height = int(W), int(H)

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
            "-s", f"{width}x{height}",
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
# Sleep Control (MediaPipe)
# ---------------------------------
class SleepControl:
    def __init__(self, **kwargs):
        self.on_sleep = []
        self.on_wake = []

        self.interval = kwargs.get("interval", 0.4)
        self.camera_id = kwargs.get("camera_id", 0)
        self.threshold = kwargs.get("threshold", 0.25)
        self.sleepsum = kwargs.get("sleepsum", 5)
        self.output_dir = resource_path(kwargs.get("output_dir", "videos"))
        self.resolution = kwargs.get("resolution", "640x480")
        self.min_recording_duration = kwargs.get("min_recording_duration", 7)

        os.makedirs(self.output_dir, exist_ok=True)

        self.judges = deque(maxlen=self.sleepsum)
        self.recording_active = False
        self.was_sleeping = False

        self.camera = CameraReader(self.camera_id, self.resolution)

        self.frame_queue = None
        self.recorder_pipe = None
        self.current_recorder = None

        self.fps = 25
        self.recording_interval = 1 / self.fps
        self.running = False

        try:
            self.mp_face = FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception as e:
            print("❌ MediaPipe init failed:", e)
            self.mp_face = None

    # EAR computation using MediaPipe indices
    def _eye_aspect_ratio(self, lm, idxs):
        p = np.array([[lm[i].x, lm[i].y] for i in idxs])
        A = np.linalg.norm(p[1] - p[5])
        B = np.linalg.norm(p[2] - p[4])
        C = np.linalg.norm(p[0] - p[3])
        return (A + B) / (2.0 * C)

    def detect_sleep(self, frame):
        if self.mp_face is None:
            return False
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.mp_face.process(rgb)

        if not res.multi_face_landmarks:
            self._handle_wake()
            return False

        lm = res.multi_face_landmarks[0].landmark

        left_eye = [33, 160, 158, 133, 153, 144]
        right_eye = [362, 385, 387, 263, 373, 380]

        ear = np.mean([
            self._eye_aspect_ratio(lm, left_eye),
            self._eye_aspect_ratio(lm, right_eye),
        ])

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

    def _start_recording(self):
        self.recording_active = True
        self.frame_queue = Queue(maxsize=100)
        parent, child = Pipe()
        self.recorder_pipe = parent

        W, H = self.resolution.split("x")
        self.current_recorder = FFmpegWriterProcess(
            self.frame_queue,
            f"{W}x{H}",
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

                if SHOW_WINDOW:
                    cv2.imshow("Webcam", frame)
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
