import threading
import os
import subprocess
import sys
import time
from time import strftime
from queue import Queue

import cv2
import flet as ft

from sleep_control import SleepControl

# ---------------------------------
# 🔧 CONFIGURATION
# ---------------------------------
config = {
    "interval": 0.4,
    "threshold": 0.25,
    "output_dir": "recordings",
    "resolution": "1280x720",
    "sleepsum": 10,
    "camera_id": 0,
    "min_recording_duration": 7
}

# ---------------------------------
# 🧾 LOGGING (UI + stdout)
# ---------------------------------
log_queue = Queue()

def log(msg):
    print(msg)
    log_queue.put(msg)

# ---------------------------------
# 🔨 UTILS
# ---------------------------------
def check_camera_access():
    cap = cv2.VideoCapture(config["camera_id"])
    time.sleep(1)
    ok = cap.isOpened()
    cap.release()
    return ok

def get_working_camera_index():
    if check_camera_access():
        return config["camera_id"]

    log("⚠️ Default camera not accessible. Scanning 0–3 …")
    for i in range(4):
        config["camera_id"] = i
        if check_camera_access():
            log(f"✅ Found working camera at index {i}")
            return i

    log("❌ No working camera found.")
    sys.exit(1)

# ---------------------------------
# 📺 SCREEN RECORDER (FFmpeg)
# ---------------------------------
class ScreenRecorder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.proc = None
        self.lock = threading.Lock()
        self.start_time = 0
        self.file = None

    def start(self):
        with self.lock:
            if self.proc is not None:
                return

            self.start_time = time.time()
            os.makedirs(self.cfg["output_dir"], exist_ok=True)

            stamp = strftime("%Y%m%d_%H%M%S")
            self.file = os.path.join(
                self.cfg["output_dir"], f"screen_{stamp}.mp4"
            )
            log_path = os.path.join(
                self.cfg["output_dir"], f"screen_{stamp}.log"
            )

            if sys.platform.startswith("linux"):
                sess = os.environ.get("XDG_SESSION_TYPE", "").lower()
                if sess == "wayland":
                    cmd = ["ffmpeg", "-y", "-f", "pipewire", "-i", "0", self.file]
                else:
                    try:
                        xr = subprocess.check_output(
                            ["xrandr"], stderr=subprocess.DEVNULL
                        ).decode()
                        res = next(l for l in xr.splitlines() if "*" in l).split()[0]
                    except Exception:
                        res = self.cfg["resolution"]

                    w, h = res.split("x")
                    disp = os.environ.get("DISPLAY", ":0")
                    cmd = [
                        "ffmpeg", "-y",
                        "-f", "x11grab",
                        "-s", f"{w}x{h}",
                        "-r", "24",
                        "-i", f"{disp}+0,0",
                        "-c:v", "libx264",
                        "-preset", "ultrafast",
                        "-pix_fmt", "yuv420p",
                        self.file
                    ]

            elif sys.platform == "darwin":
                cmd = [
                    "ffmpeg", "-y",
                    "-f", "avfoundation",
                    "-framerate", "30",
                    "-i", "1:none",
                    self.file
                ]

            elif sys.platform.startswith("win"):
                cmd = [
                    "ffmpeg", "-y",
                    "-f", "gdigrab",
                    "-framerate", "30",
                    "-i", "desktop",
                    self.file
                ]
            else:
                raise RuntimeError("Unsupported OS")

            with open(log_path, "w") as logf:
                self.proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=logf,
                    stderr=logf
                )

            log(f"🖥️ Screen recording STARTED → {self.file}")

    def stop(self):
        with self.lock:
            if self.proc is None:
                return

            wait_for = self.cfg["min_recording_duration"]
            log(f"⏳ Waiting {wait_for}s to align with webcam file …")
            time.sleep(wait_for)

            log("🛑 Stopping screen recording …")
            try:
                self.proc.stdin.write(b"q\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.proc.kill()

            log(f"✅ Screen recording SAVED → {self.file}")
            self.proc = None

# Global instance
screen_recorder = ScreenRecorder(config)

# ---------------------------------
# 🔌 CALLBACKS
# ---------------------------------
def start_screen_recording():
    screen_recorder.start()

def stop_screen_recording():
    screen_recorder.stop()

# ---------------------------------
# 🎥 WEBCAM CONTROL
# ---------------------------------
def start_capturing():
    if not webcam_service.running:
        threading.Thread(
            target=webcam_service.start_capturing,
            daemon=True
        ).start()
        log("🟢 Webcam service started")

def stop_capturing():
    if webcam_service.running:
        webcam_service.stop_capturing()
        log("🔴 Webcam service stopped")

# ---------------------------------
# 🖥️ FLET UI
# ---------------------------------
def main(page: ft.Page):
    page.title = "Sleep Control Monitor"
    page.window_width = 420
    page.window_height = 300
    page.window_resizable = False

    status = ft.Text("🔄 Initializing…", size=12)
    log_view = ft.TextField(
        multiline=True,
        read_only=True,
        expand=True,
        text_size=12,
        border=ft.InputBorder.OUTLINE,
    )

    page.add(status, log_view)

    def log_pump():
        while True:
            msg = log_queue.get()
            log_view.value += msg + "\n"
            log_view.cursor_position = len(log_view.value)
            page.update()

    threading.Thread(target=log_pump, daemon=True).start()

    try:
        get_working_camera_index()
        status.value = "📷 Camera OK"
        page.update()

        global webcam_service
        webcam_service = SleepControl(**config)

        webcam_service.on_sleep.append(start_screen_recording)
        webcam_service.on_wake.append(stop_screen_recording)

        start_capturing()

    except Exception as e:
        log(f"❌ Startup error: {e}")
        status.value = "❌ Startup failed"
        page.update()

# ---------------------------------
# 🚀 ENTRY POINT
# ---------------------------------
if __name__ == "__main__":
    ft.run(main)