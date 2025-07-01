import eel
import threading
import os
import subprocess
import sys
from sleep_control import SleepControl
from time import sleep, strftime
import cv2

# Configuration
config = {
    "interval": 0.4,
    "threshold": 0.25,
    "output_dir": "recordings",
    "resolution": "1280x720",  # Reduced resolution for testing
    "sleepsum": 10,  # the amount of frames that calculates the average of sleep
    "camera_id": 0,  # Try different indices if needed
}

# Globals for screen recording
_screen_proc = None
_screen_lock = threading.Lock()

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath('.'), relative_path)

def check_camera_access():
    """Check if camera is accessible"""
    cap = cv2.VideoCapture(config["camera_id"])
    sleep(1)
    if cap.isOpened():
        cap.release()
        return True
    return False

def get_working_camera_index():
    if check_camera_access():
        return config["camera_id"]

    print("⚠️ Camera not accessible. Trying alternative solutions...")
    for i in range(0, 4):
        config["camera_id"] = i
        if check_camera_access():
            print(f"✅ Found working camera at index {i}")
            return i
    print("❌ No working camera found. Please check your camera connection.")
    exit(1)

def start_screen_recording():
    import subprocess
    global _screen_proc
    with _screen_lock:
        if _screen_proc is not None:
            return  # already recording

        os.makedirs(config['output_dir'], exist_ok=True)
        timestamp = strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(config['output_dir'], f"screen_{timestamp}.mp4")
        log_path = os.path.join(config['output_dir'], f"screen_{timestamp}.log")

        if sys.platform.startswith("linux"):
            session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()

            if session_type == "wayland":
                # PipeWire (Wayland)
                cmd = [
                    "ffmpeg", "-y",
                    "-f", "pipewire",
                    "-i", "0",
                    filename
                ]
            else:
                # X11 - Auto-detect resolution
                try:
                    xrandr_output = subprocess.check_output(
                        ["xrandr"], stderr=subprocess.DEVNULL
                    ).decode()
                    # Find the line with '*' indicating the current mode
                    for line in xrandr_output.splitlines():
                        if "*" in line:
                            resolution_str = line.strip().split()[0]
                            break
                    else:
                        # fallback to config if not found
                        resolution_str = config["resolution"]
                    width, height = resolution_str.split("x")
                except Exception as e:
                    print(f"⚠️ Could not detect screen resolution: {e}")
                    width, height = config["resolution"].split("x")

                display = os.environ.get("DISPLAY", ":0")

                # Use -s instead of -video_size to match your working example
                cmd = [
                    "ffmpeg", "-y",
                    "-f", "x11grab",
                    "-s", f"{width}x{height}",
                    "-r", "24",
                    "-i", f"{display}+0,0",
                    "-c:v", "libx264",
                    "-preset", "ultrafast",
                    "-pix_fmt", "yuv420p",
                    filename
                ]

        elif sys.platform == "darwin":
            # macOS
            cmd = [
                "ffmpeg", "-y",
                "-f", "avfoundation",
                "-framerate", "30",
                "-i", "1:none",
                filename
            ]

        elif sys.platform.startswith("win"):
            # Windows
            cmd = [
                "ffmpeg", "-y",
                "-f", "gdigrab",
                "-framerate", "30",
                "-i", "desktop",
                filename
            ]
        else:
            raise RuntimeError("Unsupported platform")

        with open(log_path, "w") as logf:
            _screen_proc = subprocess.Popen(cmd, stdout=logf, stderr=logf)
        print(f"🖥️ Started screen recording with FFmpeg: {filename}\n(see {log_path} for logs)")


def stop_screen_recording():
    """Stop the ffmpeg recording process"""
    global _screen_proc
    with _screen_lock:
        if _screen_proc is not None:
            _screen_proc.terminate()
            _screen_proc.wait()
            print("🛑 Screen recording stopped")
            _screen_proc = None

def start_capturing():
    if not webcam_service.running:
        thread = threading.Thread(
            target=webcam_service.start_capturing,
            daemon=True,
            name="WebcamCaptureThread"
        )
        thread.start()
        print("🟢 Webcam service started")

def stop_capturing():
    if webcam_service.running:
        webcam_service.stop_capturing()
        print("🔴 Webcam service stopped")

@eel.expose
def get_camera_status():
    return {
        "connected": check_camera_access(),
        "index": config["camera_id"],
        "resolution": config["resolution"]
    }

# ------------------------------
# ✅ MacOS-Safe Entry Point
# ------------------------------
if __name__ == "__main__":
    get_working_camera_index()
    webcam_service = SleepControl(**config)

    # Hook into sleep/wake events
    try:
        webcam_service.on_sleep.append(start_screen_recording)
        webcam_service.on_wake.append(stop_screen_recording)
    except AttributeError:
        print("⚠️ SleepControl has no event hooks; ensure it supports on_sleep/on_wake lists.")

    eel.init(resource_path("web"))

    try:
        start_capturing()
        eel.start("index.html", size=(300, 300), block=True, mode='chrome')
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    finally:
        stop_capturing()
        stop_screen_recording()
