import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR.parent / "models" / "yolov8s.pt"

YOLO_MODEL_PATH = Path(os.environ.get("YOLO_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
YOLO_DEVICE = os.environ.get("YOLO_DEVICE", "auto")

CAMERA_CONF = float(os.environ.get("CAMERA_CONF", "0.22"))
CAMERA_IOU = float(os.environ.get("CAMERA_IOU", "0.50"))
CAMERA_IMGSZ = int(os.environ.get("CAMERA_IMGSZ", "960"))
CAMERA_MIN_AREA = int(os.environ.get("CAMERA_MIN_AREA", "35"))
