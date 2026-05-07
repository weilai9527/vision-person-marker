@echo off
setlocal

set "PROJECT_DIR=D:\vision-person-marker\yolo-person-web"
set "PYTHON_EXE=D:\python\python.exe"
set "DATABASE_URL=sqlite:///D:/vision-person-marker/yolo-person-web/database/person_marker.db"
set "LLM_TIMEOUT=180"
set "MODEL_IMAGE_MAX_SIDE=0"
set "BOX_EXPAND_X=0"
set "BOX_EXPAND_Y=0"
set "ENABLE_TILE_DETECTION=0"
set "TILE_GRID=2"
set "TILE_OVERLAP=0.18"
set "YOLO_MODEL_PATH=D:\vision-person-marker\models\yolov8s.pt"
set "YOLO_CONF=0.25"
set "YOLO_IOU=0.50"
set "YOLO_IMGSZ=1280"
set "YOLO_MIN_AREA=60"

cd /d "%PROJECT_DIR%"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python not found:
    echo %PYTHON_EXE%
    echo.
    echo Please check the virtual environment path in start.bat.
    pause
    exit /b 1
)

if not exist "app.py" (
    echo [ERROR] app.py not found in:
    echo %PROJECT_DIR%
    pause
    exit /b 1
)

echo Starting yolo-person-web...
echo URL: http://127.0.0.1:5000
echo.

for /f "tokens=5" %%P in ('netstat -ano ^| findstr "127.0.0.1:5000" ^| findstr "LISTENING"') do (
    echo Stopping old server process %%P on port 5000...
    taskkill /PID %%P /F >nul 2>nul
)

echo.

"%PYTHON_EXE%" app.py

echo.
echo Server stopped.
pause
