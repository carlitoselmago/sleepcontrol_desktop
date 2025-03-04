import cv2
import time
from datetime import datetime

# Configuration
INTERVAL_SECONDS = 10  # Set the interval between captures (e.g., 10 seconds)
OUTPUT_DIR = ""        # Leave empty to save in current directory

def capture_webcam():
    cap = cv2.VideoCapture(0)
    try:
        if not cap.isOpened():
            return False
        ret, frame = cap.read()
        if ret:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"photos/{OUTPUT_DIR}webcam_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
        return ret
    finally:
        cap.release()

if __name__ == "__main__":
    while True:
        success = capture_webcam()
        status = f"{datetime.now()} - {'Captured' if success else 'Failed'}"
        with open("webcam_log.txt", "a") as f:
            f.write(status + "\n")
        time.sleep(INTERVAL_SECONDS)