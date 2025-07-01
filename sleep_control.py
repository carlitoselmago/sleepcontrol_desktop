import cv2
import time
import os
import dlib
import numpy as np
from datetime import datetime
from threading import Thread, Lock
from collections import deque
from multiprocessing import Process, Queue, Pipe
import platform
import subprocess
import sys

try:
    import psutil
except ImportError:
    psutil = None

SHOW_WINDOW = False

# Helper for PyInstaller resource loading
def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath('.'), relative_path)

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
        def set_linux_auto_exposure(cam_id):
            import shutil
            if shutil.which("v4l2-ctl") is None:
                print("⚠️ v4l2-ctl not found. Skipping auto exposure config.")
                return
            try:
                device = f"/dev/video{cam_id}"
                subprocess.run(["v4l2-ctl", "--device", device, "-c", "exposure_auto=1"], check=True)
                print("🌞 Auto exposure enabled via v4l2-ctl")
            except Exception as e:
                print("⚠️ Failed to set auto exposure via v4l2-ctl:", e)

        backend = cv2.CAP_V4L2 if platform.system() != "Darwin" else cv2.CAP_AVFOUNDATION
        self.cap = cv2.VideoCapture(self.camera_id, backend)
        if self.cap.isOpened():
            W, H = self.resolution.split("x")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(W))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(H))
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            if platform.system() != "Darwin":
                set_linux_auto_exposure(self.camera_id)
            print(f"✅ Camera opened successfully with backend {backend}")
        else:
            print("❌ Failed to open camera")
            self.running = False

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                print("⚠️ Camera read failed")
                time.sleep(0.1)

    def get_frame(self):
        with self.lock:
            if self.frame is not None:
                frame_copy = self.frame.copy()
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame_copy, timestamp, (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 1)
                return frame_copy
            return None

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()

class FFmpegWriterProcess(Process):
    def __init__(self, frame_queue, resolution, fps, output_path, control_pipe):
        super().__init__()
        self.frame_queue = frame_queue
        self.resolution = resolution
        self.fps = fps
        self.output_path = output_path
        self.control_pipe = control_pipe

    def run(self):
        print('📹 FFmpeg writer process started')
        if psutil and platform.system() != "Darwin":
            try:
                psutil.Process().cpu_affinity([1])
                print("📍 Writer bound to core 1")
            except Exception as e:
                print("⚠️ CPU affinity error:", e)

        W, H = self.resolution.split("x")
        width, height = int(W), int(H)
        filename = os.path.join(self.output_path, f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")

        frame_timestamps = []
        raw_frames = []
        last_activity = time.time()
        min_duration = 7

        while True:
            if self.control_pipe.poll():
                msg = self.control_pipe.recv()
                if msg == "extend":
                    last_activity = time.time()
                elif msg == "stop":
                    break
            if time.time() - last_activity > min_duration:
                break
            try:
                frame = self.frame_queue.get(timeout=0.5)
                raw_frames.append(frame)
                frame_timestamps.append(time.time())
            except Exception:
                continue

        if len(frame_timestamps) >= 2:
            total_time = frame_timestamps[-1] - frame_timestamps[0]
            actual_fps = max(1, len(frame_timestamps) / total_time)
        else:
            actual_fps = self.fps

        print(f"📹 Saving video (~{actual_fps:.2f} FPS)")
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f'{width}x{height}', '-r', str(actual_fps), '-i', '-',
            '-an', '-vcodec', 'libx264', '-preset', 'ultrafast',
            '-pix_fmt', 'yuv420p', '-r', str(actual_fps), filename
        ]

        if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
            ffmpeg_cmd.insert(0, resource_path("ffmpeg"))
        else:
            print('💻 Running in normal Python process')

        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
        try:
            for frame in raw_frames:
                proc.stdin.write(frame.tobytes())
            proc.stdin.close()
            proc.wait()
            print(f"✅ FFmpeg writer finished. Frames: {len(raw_frames)}")
        except Exception as e:
            print("⚠️ Error writing video:", e)

class SleepControl:
    def __init__(self, **kwargs):
        # Event hooks
        self.on_sleep = []
        self.on_wake = []

        # Configs
        self.interval = kwargs.get("interval", 0.4)
        self.camera_id = kwargs.get("camera_id", 0)
        self.threshold = kwargs.get("threshold", 0.25)
        self.sleepsum = kwargs.get("sleepsum", 3)
        self.output_dir = resource_path(kwargs.get("output_dir", "photos"))
        os.makedirs(self.output_dir, exist_ok=True)
        self.resolution = kwargs.get("resolution", "640x480")
        self.log_file = kwargs.get("log_file", "webcam.log")
        self.predictor_path = resource_path(kwargs.get("predictor_path", "data/shape_predictor_68_face_landmarks.dat"))

        # dlib face detection
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.predictor_path)

        # State
        self.judges = deque(maxlen=self.sleepsum)
        self.scale_factor = 0.5
        self.fps = 25
        self.recording_interval = 1.0 / self.fps
        self.recording_active = False
        self.current_recorder = None
        self.frame_queue = None
        self.recorder_pipe = None
        self.running = False
        self.was_sleeping = False

        # Camera
        W, H = self.resolution.split("x")
        self.width, self.height = int(W), int(H)
        self.camera = CameraReader(self.camera_id, self.resolution)

    def _write_log(self, message):
        with open(self.log_file, "a") as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")

    def detect_sleep(self, frame):
        # Face landmarks -> eye aspect ratio
        small = cv2.resize(frame, (0,0), fx=self.scale_factor, fy=self.scale_factor)
        gray_s = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray_s, 1)
        if not rects:
            self._handle_wake()
            return False
        # Scale rectangle
        r = rects[0]
        rect = dlib.rectangle(
            int(r.left()/self.scale_factor), int(r.top()/self.scale_factor),
            int(r.right()/self.scale_factor), int(r.bottom()/self.scale_factor)
        )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lm = self.predictor(gray, rect)
        landmarks = np.array([[p.x,p.y] for p in lm.parts()])
        ear = np.mean([
            np.linalg.norm(landmarks[37]-landmarks[41]),
            np.linalg.norm(landmarks[38]-landmarks[40]),
            np.linalg.norm(landmarks[43]-landmarks[47]),
            np.linalg.norm(landmarks[44]-landmarks[46])
        ]) / np.mean([
            np.linalg.norm(landmarks[36]-landmarks[39]),
            np.linalg.norm(landmarks[42]-landmarks[45])
        ])
        self.judges.append(ear)
        avg = np.mean(self.judges)
        print("👁️ Sleep judge mean:", avg)
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
        # Initialize queue & process
        self.recording_active = True
        self.frame_queue = Queue(maxsize=100)
        parent, child = Pipe()
        self.recorder_pipe = parent
        self.current_recorder = FFmpegWriterProcess(
            self.frame_queue, f"{self.width}x{self.height}", self.fps, self.output_dir, child
        )
        self.current_recorder.start()
        self.was_sleeping = True
        self._write_log("Started recording")
        # Fire sleep hooks
        for cb in self.on_sleep:
            try: cb()
            except Exception as e: print("⚠️ on_sleep hook error:", e)

    def _handle_wake(self):
        if self.was_sleeping:
            # Fire wake hooks
            for cb in self.on_wake:
                try: cb()
                except Exception as e: print("⚠️ on_wake hook error:", e)
            self.was_sleeping = False
        # Let the recorder finish based on its own min_duration

    def _check_recorder(self):
        if self.current_recorder and not self.current_recorder.is_alive():
            self.recording_active = False
            self.current_recorder = None
            self.frame_queue = None
            self.recorder_pipe = None
            self._write_log("Recording finished")
            print("📼 Recording process ended")

    def start_capturing(self):
        self._write_log("Capture started")
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
                if self.recording_active and self.frame_queue and not self.frame_queue.full() and now - last_frame >= self.recording_interval:
                    self.frame_queue.put(frame)
                    last_frame = now
                self._check_recorder()
                if SHOW_WINDOW:
                    cv2.imshow("Webcam", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                time.sleep(0.01)
        finally:
            self.stop_capturing()

    def stop_capturing(self):
        self.running = False
        if self.current_recorder:
            self.recorder_pipe.send("stop")
            self.current_recorder.join()
        self.camera.stop()
        self._write_log("Capture stopped")
        if SHOW_WINDOW:
            cv2.destroyAllWindows()
