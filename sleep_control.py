import cv2
import time
import os
import dlib
import numpy as np
from datetime import datetime
from threading import Thread

SHOW_WINDOW = False  # Global variable to control window display

class SleepControl:
    def __init__(
        self,
        interval: float = 0.1,  # 0.1 second interval
        camera_id: int = 0,
        threshold: float 0.25, # cantidad de ojo cerrado de 0 a 1
        sleepsum: int 3, # cantidad de frames en los que se cumple la condicion de threshold
        output_dir: str = "photos",
        resolution: str= "640x480",
        log_file: str = "webcam.log",
        predictor_path: str = "data/shape_predictor_68_face_landmarks.dat"
    ):
        self.interval = interval
        self.camera_id = camera_id
        self.sleepsum = sleepsum
        self.threshold = threshold
        self.output_dir = output_dir
        self.log_file = log_file
        self.resolution = resolution
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.flags = []  # List to store the last flags
        self.running = False
        self._create_output_dir()
        self.scale_factor = 0.5  # For optimized face detection

    def _create_output_dir(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

    def _write_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "a") as f:
            f.write(f"[{timestamp}] {message}\n")

    def _save_image(self, frame: np.ndarray, filename: str) -> None:
        """Saves the image in a separate thread."""
        def save():
            cv2.imwrite(filename, frame)
            self._write_log(f"Saved image: {filename}")
        
        Thread(target=save, daemon=True).start()

    def detect_sleep(self, frame) -> bool:
        # Optimized face detection on smaller frame
        small_frame = cv2.resize(frame, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
        gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray_small, 1)

        if len(rects) == 0:
            return False  # No face detected

        # Scale face rectangle back to original size
        rect = rects[0]
        original_rect = dlib.rectangle(
            int(rect.left() / self.scale_factor),
            int(rect.top() / self.scale_factor),
            int(rect.right() / self.scale_factor),
            int(rect.bottom() / self.scale_factor)
        )

        # Landmark detection on original frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        landmarks = self.predictor(gray, original_rect)
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

        # Eye aspect ratio calculation
        d1 = np.linalg.norm(landmarks[37] - landmarks[41])
        d2 = np.linalg.norm(landmarks[38] - landmarks[40])
        d3 = np.linalg.norm(landmarks[43] - landmarks[47])
        d4 = np.linalg.norm(landmarks[44] - landmarks[46])
        d_mean = (d1 + d2 + d3 + d4) / 4

        d5 = np.linalg.norm(landmarks[36] - landmarks[39])
        d6 = np.linalg.norm(landmarks[42] - landmarks[45])
        d_reference = (d5 + d6) / 2
        d_judge = d_mean / d_reference

        flag = int(d_judge < self.threshold)  # Threshold for closed eyes

        # Store the last 30 flags in a list
        self.flags.append(flag)
        if len(self.flags) > self.sleepsum:
            self.flags.pop(0)

        # Check if the majority of the last 30 frames indicate sleep
        if sum(self.flags) > len(self.flags) / 2 and flag == 1:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f"webcam_{timestamp}.jpg")
            print("Save image")
            self._save_image(frame.copy(), filename)  # Save image in a separate thread

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

        # Set camera to lower resolution for faster processing
        W,H=self.resolution.split("x")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(W))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(H))

        self.running = True
        try:
            while self.running:
                start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    self._write_log(f"Failed to capture from camera ID {self.camera_id}")
                    continue

                self.detect_sleep(frame)

                if SHOW_WINDOW:
                    cv2.imshow("Capture", frame)
                    cv2.waitKey(1)

                # Maintain accurate interval timing
                processing_time = time.time() - start_time
                sleep_time = self.interval - processing_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    self._write_log(f"Can't keep up with interval! Processing took {processing_time:.2f}s")

        except Exception as e:
            self._write_log(f"Unexpected error: {str(e)}")
        finally:
            cap.release()
            if SHOW_WINDOW:
                cv2.destroyAllWindows()

    def stop_capturing(self):
        self.running = False
        self._write_log("Service stopped")