rm -r dist
# Build with PyInstaller
pyinstaller --noconsole --add-data "web:./web" --hidden-import eel main.py

cp -r data dist/data

# Add camera permission to Info.plist
/usr/libexec/PlistBuddy -c "Add :NSCameraUsageDescription string 'This app uses the camera to detect sleep for recording'" dist/main.app/Contents/Info.plist