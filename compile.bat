@echo off

REM Remove the "dist" directory if it exists
if exist dist rmdir /s /q dist

REM Run PyInstaller to create the executable
pyinstaller --noconsole --onefile --add-data "web;web" --icon=icon.png --hidden-import eel main.py

REM Copy the "data" folder to the "dist" directory
xcopy /e /i /y data dist\data

echo Build complete.
pause
