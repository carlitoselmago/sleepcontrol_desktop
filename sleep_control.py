import cv2
import time
import os
import dlib
import numpy as np
from datetime import datetime
from threading import Thread, Lock, Event
from collections import deque
import psutil
import subprocess
import platform

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
        self.ready = Event()
        self.initialize_camera()

    def initialize_camera(self):
        backends = [cv2.CAP_V4L2, cv2.CAP_DSHOW, cv2.CAP_ANY]
        for backend in backends:
            self.cap = cv2.VideoCapture(self.camera_id, backend)
            if self.cap.isOpened():
                W, H = self.resolution.split("x")
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(W))
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(H))
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                print(f"✅ Camera opened successfully with backend {backend}")
                self.ready.set()
                return

        print("❌ Failed to open camera with all backends tried.")
        self.running = False

    def run(self):
        if platform.system() != 'Darwin':  # 'Darwin' is macOS
            try:
                psutil.Process().cpu_affinity([0])
            except AttributeError:
                print("cpu_affinity not available on this platform.")
            except Exception as e:
                print(f"Could not set CPU affinity: {e}")

        while self.running:
            if not self.ready.is_set():
                time.sleep(0.1)
                continue

            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                print("Camera read failed, attempting to reinitialize...")
                self.cap.release()
                time.sleep(1)
                self.initialize_camera()

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
        if self.cap is not None:
            self.cap.release()

class FFmpegRecorder(Thread):
    def __init__(self, get_frame_callback, width, height, fps, output_path, duration):
        super().__init__()
        self.get_frame = get_frame_callback
        self.width = width
        self.height = height
        self.fps = fps
        self.output_path = output_path
        self.duration = duration
        self.stop_event = Event()

    def run(self):
        filename = os.path.join(
            self.output_path,
            f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        )

        ffmpeg_cmd = [
            'ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{self.width}x{self.height}',
            '-r', str(self.fps),
            '-i', '-',
            '-an',
            '-vcodec', 'libx264',
            '-pix_fmt', 'yuv420p',
            filename
        ]

        process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
        frame_duration = 1.0 / self.fps
        start_time = time.time()

        while not self.stop_event.is_set() and (time.time() - start_time < self.duration):
            frame = self.get_frame()
            if frame is not None:
                try:
                    process.stdin.write(frame.tobytes())
                except Exception as e:
                    print("FFmpeg write error:", e)
                    break
            time.sleep(frame_duration)

        try:
            process.stdin.close()
            process.wait()
        except Exception as e:
            print("FFmpeg process close error:", e)

    def stop(self):
        self.stop_event.set()

class SleepControl:
    def __init__(self, **kwargs):
        self.interval = kwargs.get("interval", 0.1)
        self.camera_id = kwargs.get("camera_id", 0)
        self.threshold = kwargs.get("threshold", 0.25)
        self.sleepsum = kwargs.get("sleepsum", 3)
        self.output_dir = kwargs.get("output_dir", "photos")
        self.resolution = kwargs.get("resolution", "640x480")
        self.log_file = kwargs.get("log_file", "webcam.log")
        self.predictor_path = kwargs.get("predictor_path", "data/shape_predictor_68_face_landmarks.dat")

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.predictor_path)

        self.judges = []
        self.scale_factor = 0.5

        self.fps = 25
        self.min_record_duration = 7
        self.prebuffer = deque(maxlen=self.fps * 2)
        self.recording_active = False
        self.current_recorder = None
        self.running = False

        W, H = self.resolution.split("x")
        self.width = int(W)
        self.height = int(H)

        self.camera = CameraReader(self.camera_id, self.resolution)
        self._create_output_dir()

    def _create_output_dir(self):
        os.makedirs(self.output_dir, exist_ok=True)

    def _write_log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "a") as f:
            f.write(f"[{timestamp}] {message}\n")

    def _draw_timestamp(self, frame):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 1)

    def detect_sleep(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
        gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray_small, 1)

        if not rects:
            return False

        rect = rects[0]
        rect = dlib.rectangle(
            int(rect.left() / self.scale_factor),
            int(rect.top() / self.scale_factor),
            int(rect.right() / self.scale_factor),
            int(rect.bottom() / self.scale_factor)
        )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        landmarks = self.predictor(gray, rect)
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

        d1 = np.linalg.norm(landmarks[37] - landmarks[41])
        d2 = np.linalg.norm(landmarks[38] - landmarks[40])
        d3 = np.linalg.norm(landmarks[43] - landmarks[47])
        d4 = np.linalg.norm(landmarks[44] - landmarks[46])
        d_mean = (d1 + d2 + d3 + d4) / 4

        d5 = np.linalg.norm(landmarks[36] - landmarks[39])
        d6 = np.linalg.norm(landmarks[42] - landmarks[45])
        d_reference = (d5 + d6) / 2
        d_judge = d_mean / d_reference

        self.judges.append(d_judge)
        if len(self.judges) > self.sleepsum:
            self.judges.pop(0)

        judge_mean = np.mean(self.judges)
        print("judge mean", judge_mean)

        if judge_mean < self.threshold and not self.recording_active:
            self.recording_active = True
            self.current_recorder = FFmpegRecorder(
                self.camera.get_frame,
                self.width,
                self.height,
                self.fps,
                self.output_dir,
                self.min_record_duration
            )
            self.current_recorder.start()
            self._write_log("Started recording via FFmpeg due to sleep detection")

        return d_judge < self.threshold


    def start_capturing(self):
        self._write_log("Webcam capture service started")
        self.camera.start()
        self.running = True

        if platform.system() != 'Darwin':  # macOS doesn't support cpu_affinity
            try:
                psutil.Process().cpu_affinity([2])
            except AttributeError:
                print("cpu_affinity not available on this platform.")
            except Exception as e:
                print(f"Could not set CPU affinity: {e}")

        try:
            while self.running:
                start_time = time.time()
                frame = self.camera.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                frame_copy = frame.copy()
                self._draw_timestamp(frame_copy)
                self.prebuffer.append(frame_copy)

                self.detect_sleep(frame_copy)

                if SHOW_WINDOW:
                    cv2.imshow("Webcam", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                elapsed = time.time() - start_time
                to_sleep = self.interval - elapsed
                if to_sleep > 0:
                    time.sleep(to_sleep)
        finally:
            self.stop_capturing()

    def stop_capturing(self):
        self.running = False
        if self.current_recorder:
            self.current_recorder.stop()
            self.current_recorder.join()
        self.camera.stop()
        self._write_log("Capture stopped")
        if SHOW_WINDOW:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = SleepControl()
    try:
        controller.start_capturing()
    except KeyboardInterrupt:
        controller.stop_capturing()
