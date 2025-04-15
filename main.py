import eel
import threading
import os
from sleep_control import SleepControl
from time import sleep
import cv2
import sys

# Configuration
config = {
    "interval": 0.4,
    "threshold": 0.25,
    "output_dir": "recordings",
    "resolution": "1280x720",  # Reduced resolution for testing
    "sleepsum": 10,  # the amount of frames that calculates the average of sleep
    "camera_id": 0,  # Try different indices if needed
}

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
def resource_path(relative_path):
        """ Get absolute path to resource, works for dev and for PyInstaller """
        if hasattr(sys, '_MEIPASS'):
            return os.path.join(sys._MEIPASS, relative_path)  # PyInstaller's temp folder
        return os.path.join(os.path.abspath("."), relative_path)
# ------------------------------
# ✅ MacOS-Safe Entry Point
# ------------------------------
if __name__ == "__main__":
    get_working_camera_index()
    webcam_service = SleepControl(**config)

    eel.init(resource_path("web"))

    try:
        start_capturing()
        eel.start("index.html", size=(300, 300), block=True, mode='chrome')
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    finally:
        stop_capturing()
