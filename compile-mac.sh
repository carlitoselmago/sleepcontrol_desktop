rm -r dist
pyinstaller --noconsole --onefile --add-data "web:./web"  --hidden-import eel main.py
cp -r data dist/data

