from sleep_control import SleepControl

if __name__ == "__main__":
    # Configuration
    config = {
        "interval": 5,          # Seconds between captures
        "output_dir": "photos"
    }

    # Initialize and start service
    webcam_service = SleepControl(**config)
    
    try:
        webcam_service.start_capturing()
    except KeyboardInterrupt:
        print("\nService stopped successfully")