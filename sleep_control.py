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
            except Exception as e:
                print("⚠️ Failed to set auto exposure via v4l2-ctl:", e)
        backend = cv2.CAP_V4L2 if platform.system() != "Darwin" else cv2.CAP_AVFOUNDATION
        self.cap = cv2.VideoCapture(self.camera_id, backend)
        if self.cap.isOpened():
            W, H = self.resolution.split("x")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(W))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(H))
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            # Enable auto exposure if supported (Linux only)
            if platform.system() != "Darwin":
                set_linux_auto_exposure(self.camera_id)
            print(f"✅ Camera opened successfully with backend {backend}")
        else:
            print("❌ Failed to open camera")
            self.running = False

    def resource_path(relative_path):
        import sys
        if hasattr(sys, '_MEIPASS'):
            return os.path.join(sys._MEIPASS, relative_path)
        return os.path.join(os.path.abspath("."), relative_path)

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

        ffmpeg_path = resource_path("ffmpeg")
        ffmpeg_cmd = [ffmpeg_path,
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
