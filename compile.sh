rm -r dist
pyinstaller --noconsole --onefile --add-data "web:./web" --icon=icon.png --hidden-import eel main.py
cp -r data dist/data

# Linux only
#gvfs-set-attribute dist/main metadata::custom-icon icon.png