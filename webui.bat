@echo off
cd /d %~dp0    

echo Installing requirements...

pip install ultralytics gradio

echo Launching app.py...

python app.py