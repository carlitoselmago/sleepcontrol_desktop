rm -r dist
# Build with PyInstaller
# pyinstaller --noconsole --onefile --add-data "web:./web" --hidden-import eel main.py
pyinstaller --add-data "web:./web" --hidden-import eel main.py

cp -r data dist/main/_internal/data

cp /opt/anaconda3/envs/sleep/bin/ffmpeg dist/main/_internal/ffmpeg
chmod +x dist/main/_internal/ffmpeg


# Add camera permission to Info.plist
/usr/libexec/PlistBuddy -c "Add :NSCameraUsageDescription string 'This app uses the camera to detect sleep for recording'" dist/main.app/Contents/Info.plist