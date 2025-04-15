rm -r dist
pyinstaller --noconsole --add-data "web:./web" \
  --hidden-import eel \
  --osx-bundle-identifier com.yourdomain.main \
  --plist=custom_Info.plist \
  main.py

cp -r data dist/data

