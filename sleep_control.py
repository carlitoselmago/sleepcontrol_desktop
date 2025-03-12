import cv2
import time
import os
import dlib
import numpy as np
from datetime import datetime
from typing import Optional

SHOW_WINDOW = False  # Global variable to control window display

class SleepControl:
    def __init__(
        self,
        interval: float = 1.0,
        camera_id: int = 0,
        output_dir: str = "photos",
        log_file: str = "webcam.log",
        predictor_path: str = "data/shape_predictor_68_face_landmarks.dat"
    ):
        self.interval = interval
        self.camera_id = camera_id
        self.output_dir = output_dir
        self.log_file = log_file
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.queue = np.zeros(30, dtype=int).tolist()
        self._create_output_dir()

    def _create_output_dir(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

    def _write_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "a") as f:
            f.write(f"[{timestamp}] {message}\n")

    def detect_sleep(self, frame) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 1)
        if len(rects) == 0:
            return False  # No face detected
        
        rect = rects[0]  # Only process the first detected face
        landmarks = self.predictor(gray, rect)
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

        # Eye landmarks indices (specific points based on 68-face landmark model)
        d1 = np.linalg.norm(landmarks[37] - landmarks[41])
        d2 = np.linalg.norm(landmarks[38] - landmarks[40])
        d3 = np.linalg.norm(landmarks[43] - landmarks[47])
        d4 = np.linalg.norm(landmarks[44] - landmarks[46])
        d_mean = (d1 + d2 + d3 + d4) / 4

        d5 = np.linalg.norm(landmarks[36] - landmarks[39])
        d6 = np.linalg.norm(landmarks[42] - landmarks[45])
        d_reference = (d5 + d6) / 2
        d_judge = d_mean / d_reference
        
        flag = int(d_judge < 0.25)  # Threshold for closed eyes
        
        self.queue = self.queue[1:] + [flag]
        if sum(self.queue) > len(self.queue) / 2:
            # actual time-based sleep detected
            #self._write_log("WARNING: Possible sleep detected!")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f"webcam_{timestamp}.jpg")
            cv2.imwrite(filename, frame)

        if SHOW_WINDOW:
            for (x, y) in landmarks:
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

        return flag == 1
    
    def start_capturing(self) -> None:
        self._write_log("Webcam capture service started")
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            self._write_log(f"Failed to open camera ID {self.camera_id}")
            return
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    self._write_log(f"Failed to capture from camera ID {self.camera_id}")
                    continue
                
                if self.detect_sleep(frame):
                    # instant detection, it might be a blink
                    #self._write_log("Drowsiness detected!")
                    pass
                
                if SHOW_WINDOW:
                    cv2.imshow("Capture", frame)
                    cv2.waitKey(1)
                
                time.sleep(self.interval)
        
        except KeyboardInterrupt:
            self._write_log("Service stopped by user")
        except Exception as e:
            self._write_log(f"Unexpected error: {str(e)}")
        finally:
            cap.release()
            if SHOW_WINDOW:
                cv2.destroyAllWindows()
