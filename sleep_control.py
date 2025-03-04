import cv2
import time
import os
from datetime import datetime
from typing import Optional

class SleepControl:
    def __init__(
        self,
        interval: int = 10,
        camera_id: int = 0,
        output_dir: str = "photos",
        log_file: str = "webcam.log"
    ):
        self.interval = interval
        self.camera_id = camera_id
        self.output_dir = output_dir
        self.log_file = log_file
        self._create_output_dir()

    def _create_output_dir(self) -> None:
        """Create output directory if it doesn't exist"""
        os.makedirs(self.output_dir, exist_ok=True)

    def _write_log(self, message: str) -> None:
        """Write messages to log file with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "a") as f:
            f.write(f"[{timestamp}] {message}\n")

    def capture(self, camera_id: Optional[int] = None) -> bool:
        """
        Capture a single frame from specified camera
        Returns True if capture succeeded
        """
        cam_id = camera_id if camera_id is not None else self.camera_id
        
        try:
            cap = cv2.VideoCapture(cam_id)
            if not cap.isOpened():
                self._write_log(f"Failed to open camera ID {cam_id}")
                return False

            ret, frame = cap.read()
            if not ret:
                self._write_log(f"Failed to capture from camera ID {cam_id}")
                return False

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f"webcam_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            return True

        except Exception as e:
            self._write_log(f"Error capturing image: {str(e)}")
            return False
        finally:
            if 'cap' in locals():
                cap.release()

    def start_capturing(self) -> None:
        """Start continuous capture with specified interval"""
        self._write_log("Webcam capture service started")
        try:
            while True:
                success = self.capture()
                status = "Capture successful" if success else "Capture failed"
                self._write_log(status)
                time.sleep(self.interval)
        except KeyboardInterrupt:
            self._write_log("Service stopped by user")
        except Exception as e:
            self._write_log(f"Unexpected error: {str(e)}")