rm -r dist
pyinstaller --noconsole --add-data "web:./web"  --hidden-import eel main.py
cp -r data dist/data

