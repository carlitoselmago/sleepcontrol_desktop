import eel
import threading
from sleep_control import SleepControl

# Configuration
config = {
    "interval": 0.1,  # Seconds between captures
    "output_dir": "photos"
}

# Initialize webcam service
webcam_service = SleepControl(**config)

# Function to start capturing in a separate thread
@eel.expose
def start_capturing():
    if not webcam_service.running:
        thread = threading.Thread(target=webcam_service.start_capturing, daemon=True)
        thread.start()

# Expose functions to the Eel frontend
@eel.expose
def stop_capturing():
    webcam_service.stop_capturing

# Initialize Eel (ensure the "web" folder contains the frontend)
eel.init("web")

# Start the GUI and keep Eel running
eel.start("index.html", size=(400, 300), block=True,mode='chrome-app')
