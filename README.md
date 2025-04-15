# SleepControl desktop

Records every N seconds and checks if fatigue on a detected face

## Requirements
python 3.11
dlib (install with conda)
ffmpeg

```
conda create --name sleep python=3.11.10 dlib=19.24.0 -c conda-forge 
conda activate sleep
pip install -r requirements.txt
```

## run
```
python main.py
```

## How to compile
```
bash compile.sh
```

Then run this to update the icon (linux only)
```
sudo update-icon-caches /usr/share/icons/*
```