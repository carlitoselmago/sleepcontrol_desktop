import eel
import threading
import os
import subprocess
import sys
from sleep_control import SleepControl
import time
from time import strftime
import cv2

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
    "min_recording_duration": 7      # seconds – shared by webcam & screen
}

# ---------------------------------
# 🔨 UTILS
# ---------------------------------
def resource_path(rel):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, rel)
    return os.path.join(os.path.abspath("."), rel)

def check_camera_access():
    cap = cv2.VideoCapture(config["camera_id"])
    time.sleep(1)
    ok = cap.isOpened()
    cap.release()
    return ok

def get_working_camera_index():
    if check_camera_access():
        return config["camera_id"]
    print("⚠️ Default camera not accessible. Scanning 0-3 …")
    for i in range(4):
        config["camera_id"] = i
        if check_camera_access():
            print(f"✅ Found working camera at index {i}")
            return i
    print("❌ No working camera found.")
    sys.exit(1)

# ---------------------------------
# 📺 SCREEN RECORDER (FFmpeg)
# ---------------------------------
class ScreenRecorder:
    def __init__(self, cfg):
        self.cfg   = cfg
        self.proc  = None
        self.lock  = threading.Lock()
        self.start_time = 0
        self.file  = None

    # ---------- start ----------
    def start(self):
        with self.lock:
            if self.proc is not None:
                return                                    # already running
            self.start_time = time.time()

            os.makedirs(self.cfg["output_dir"], exist_ok=True)
            stamp      = strftime("%Y%m%d_%H%M%S")
            self.file  = os.path.join(self.cfg["output_dir"],
                                      f"screen_{stamp}.mp4")
            log_path   = os.path.join(self.cfg["output_dir"],
                                      f"screen_{stamp}.log")

            # ---- build ffmpeg cmd per-platform ----
            if sys.platform.startswith("linux"):
                sess = os.environ.get("XDG_SESSION_TYPE", "").lower()
                if sess == "wayland":
                    cmd = ["ffmpeg", "-y", "-f", "pipewire", "-i", "0",
                           self.file]
                else:  # X11
                    try:
                        xr = subprocess.check_output(["xrandr"],
                                stderr=subprocess.DEVNULL).decode()
                        res = next(l for l in xr.splitlines() if "*" in l
                                   ).strip().split()[0]
                    except Exception:
                        res = self.cfg["resolution"]
                    w, h = res.split("x")
                    disp = os.environ.get("DISPLAY", ":0")
                    cmd  = ["ffmpeg", "-y", "-f", "x11grab",
                            "-s", f"{w}x{h}", "-r", "24",
                            "-i", f"{disp}+0,0",
                            "-c:v", "libx264", "-preset", "ultrafast",
                            "-pix_fmt", "yuv420p", self.file]

            elif sys.platform == "darwin":
                cmd = ["ffmpeg", "-y", "-f", "avfoundation",
                       "-framerate", "30", "-i", "1:none", self.file]

            elif sys.platform.startswith("win"):
                cmd = ["ffmpeg", "-y", "-f", "gdigrab",
                       "-framerate", "30", "-i", "desktop", self.file]
            else:
                raise RuntimeError("Unsupported OS")

            # ---- launch ----
            with open(log_path, "w") as logf:
                self.proc = subprocess.Popen(
                    cmd, stdin=subprocess.PIPE, stdout=logf, stderr=logf
                )
            print(f"🖥️ Screen recording STARTED  →  {self.file}")

    # ---------- stop ----------
    def stop(self):
        with self.lock:
            if self.proc is None:
                return                                    # nothing to stop

            # ALWAYS wait the configured duration AFTER on_wake
            wait_for = self.cfg["min_recording_duration"]
            print(f"⏳ Waiting {wait_for}s to align with webcam file …")
            time.sleep(wait_for)

            print("🛑 Stopping screen recording …")
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

            self.proc = None
            print(f"✅ Screen recording SAVED   →  {self.file}")

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
# 🎥 WEBCAM CAPTURE WRAPPER
# ---------------------------------
def start_capturing():
    if not webcam_service.running:
        threading.Thread(target=webcam_service.start_capturing,
                         daemon=True).start()
        print("🟢 Webcam service started")

def stop_capturing():
    if webcam_service.running:
        webcam_service.stop_capturing()
        print("🔴 Webcam service stopped")

@eel.expose
def get_camera_status():
    return {"connected": check_camera_access(),
            "index":     config["camera_id"],
            "resolution":config["resolution"]}

# ---------------------------------
# 🚀 MAIN
# ---------------------------------
if __name__ == "__main__":
    get_working_camera_index()
    webcam_service = SleepControl(**config)

    # register hooks
    webcam_service.on_sleep.append(start_screen_recording)
    webcam_service.on_wake.append(stop_screen_recording)

    eel.init(resource_path("web"))

    try:
        start_capturing()
        eel.start("index.html", size=(300, 300), block=True, mode="chrome")
    except KeyboardInterrupt:
        print("\n🔚 Interrupted by user")
    finally:
        stop_capturing()
        stop_screen_recording()
