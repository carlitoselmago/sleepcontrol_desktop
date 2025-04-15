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
        backend = cv2.CAP_V4L2 if platform.system() != "Darwin" else cv2.CAP_AVFOUNDATION
        self.cap = cv2.VideoCapture(self.camera_id, backend)
        if self.cap.isOpened():
            W, H = self.resolution.split("x")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(W))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(H))
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            # Enable auto exposure if supported
        
            #https://github.com/opencv/opencv/issues/9738
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75) # 0.75 tested on macos
            print("🌞 Auto exposure enabled")
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
        last_activity_time = time.time()
        min_duration = 7

        while True:
            if self.control_pipe.poll():
                msg = self.control_pipe.recv()
                if msg == "extend":
                    last_activity_time = time.time()
                elif msg == "stop":
                    break

            if time.time() - last_activity_time > min_duration:
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

        print(f"📹 Saving video with estimated FPS: {actual_fps:.2f}")

        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{width}x{height}',
            '-r', str(actual_fps),
            '-i', '-',
            '-an',
            '-vcodec', 'libx264',
            '-preset', 'ultrafast',
            '-pix_fmt', 'yuv420p',
            '-r', str(actual_fps),
            filename
        ]

        process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

        try:
            for frame in raw_frames:
                process.stdin.write(frame.tobytes())
            process.stdin.close()
            process.wait()
            print(f"✅ FFmpeg writer process finished. Total frames: {len(raw_frames)}")
        except Exception as e:
            print("⚠️ Error closing FFmpeg:", e)

def resource_path(relative_path):
        """ Get absolute path to resource, works for dev and for PyInstaller """
        if hasattr(sys, '_MEIPASS'):
            return os.path.join(sys._MEIPASS, relative_path)  # PyInstaller's temp folder
        return os.path.join(os.path.abspath("."), relative_path)
        
class SleepControl:
    def __init__(self, **kwargs):
        self.interval = kwargs.get("interval", 0.4)
        self.camera_id = kwargs.get("camera_id", 0)
        self.threshold = kwargs.get("threshold", 0.25)
        self.sleepsum = kwargs.get("sleepsum", 3)
        self.output_dir = resource_path(kwargs.get("output_dir", "photos"))
        self.resolution = kwargs.get("resolution", "640x480")
        self.log_file = kwargs.get("log_file", "webcam.log")
        self.predictor_path = resource_path(kwargs.get("predictor_path", "data/shape_predictor_68_face_landmarks.dat"))

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.predictor_path)

        self.judges = []
        self.scale_factor = 0.5
        self.fps = 25
        self.recording_interval = 1.0 / self.fps
        self.recording_active = False
        self.current_recorder = None
        self.frame_queue = None
        self.running = False
        self.recorder_pipe = None

        W, H = self.resolution.split("x")
        self.width = int(W)
        self.height = int(H)

        self.camera = CameraReader(self.camera_id, self.resolution)
        os.makedirs(self.output_dir, exist_ok=True)
    
    

    def _write_log(self, message):
        with open(self.log_file, "a") as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")

    def detect_sleep(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
        gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray_small, 1)

        if not rects:
            return False

        rect = rects[0]
        rect = dlib.rectangle(
            int(rect.left() / self.scale_factor), int(rect.top() / self.scale_factor),
            int(rect.right() / self.scale_factor), int(rect.bottom() / self.scale_factor)
        )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        landmarks = self.predictor(gray, rect)
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

        d_mean = np.mean([
            np.linalg.norm(landmarks[37] - landmarks[41]),
            np.linalg.norm(landmarks[38] - landmarks[40]),
            np.linalg.norm(landmarks[43] - landmarks[47]),
            np.linalg.norm(landmarks[44] - landmarks[46])
        ])

        d_reference = np.mean([
            np.linalg.norm(landmarks[36] - landmarks[39]),
            np.linalg.norm(landmarks[42] - landmarks[45])
        ])

        d_judge = d_mean / d_reference
        self.judges.append(d_judge)
        if len(self.judges) > self.sleepsum:
            self.judges.pop(0)

        judge_mean = np.mean(self.judges)
        print("👁️ Sleep judge mean:", judge_mean)

        if judge_mean < self.threshold:
            if not self.recording_active:
                self.recording_active = True
                self.frame_queue = Queue(maxsize=100)
                parent_pipe, child_pipe = Pipe()
                self.recorder_pipe = parent_pipe
                self.current_recorder = FFmpegWriterProcess(
                    self.frame_queue,
                    self.resolution,
                    self.fps,
                    self.output_dir,
                    child_pipe
                )
                self.current_recorder.start()
                self._write_log("Started recording")
            else:
                if self.recorder_pipe:
                    self.recorder_pipe.send("extend")

        return d_judge < self.threshold

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

        last_detection_time = 0
        last_record_frame_time = 0

        try:
            while self.running:
                now = time.time()
                frame = self.camera.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                if now - last_detection_time >= self.interval:
                    self.detect_sleep(frame)
                    last_detection_time = now

                if self.recording_active and self.frame_queue and not self.frame_queue.full():
                    if now - last_record_frame_time >= self.recording_interval:
                        self.frame_queue.put(frame)
                        last_record_frame_time = now

                self._check_recorder()

                if SHOW_WINDOW:
                    cv2.imshow("Webcam", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                time.sleep(0.01)
        finally:
            self.stop_capturing()

    def stop_capturing(self):
        self.running = False
        if self.current_recorder:
            if self.recorder_pipe:
                self.recorder_pipe.send("stop")
            self.current_recorder.join()
        self.camera.stop()
        self._write_log("Capture stopped")
        if SHOW_WINDOW:
            cv2.destroyAllWindows()