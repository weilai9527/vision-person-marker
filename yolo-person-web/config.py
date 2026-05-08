import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATABASE_DIR = BASE_DIR / "instance"
DATABASE_PATH = DATABASE_DIR / "person_marker.db"
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
RESULT_DIR = BASE_DIR / "static" / "results"
LOG_DIR = BASE_DIR / "instance" / "logs"
API_CONFIG_PATH = BASE_DIR / "instance" / "api_config.json"

LLM_API_URL = os.environ.get("LLM_API_URL", "https://api.openai.com/v1/chat/completions")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "180"))
MODEL_IMAGE_MAX_SIDE = int(os.environ.get("MODEL_IMAGE_MAX_SIDE", "0"))
MODEL_IMAGE_JPEG_QUALITY = int(os.environ.get("MODEL_IMAGE_JPEG_QUALITY", "82"))
MODEL_IMAGE_MIN_SIDE = int(os.environ.get("MODEL_IMAGE_MIN_SIDE", "900"))
MODEL_IMAGE_MAX_PIXELS = int(os.environ.get("MODEL_IMAGE_MAX_PIXELS", "1600000"))
BOX_EXPAND_X = float(os.environ.get("BOX_EXPAND_X", "0"))
BOX_EXPAND_Y = float(os.environ.get("BOX_EXPAND_Y", "0"))
ENABLE_TILE_DETECTION = os.environ.get("ENABLE_TILE_DETECTION", "0") == "1"
TILE_GRID = int(os.environ.get("TILE_GRID", "2"))
TILE_OVERLAP = float(os.environ.get("TILE_OVERLAP", "0.18"))
NMS_IOU_THRESHOLD = float(os.environ.get("NMS_IOU_THRESHOLD", "0.45"))
YOLO_MODEL_PATH = Path(os.environ.get("YOLO_MODEL_PATH", str(BASE_DIR.parent / "models" / "yolov8s.pt")))
YOLO_CONF = float(os.environ.get("YOLO_CONF", "0.25"))
YOLO_IOU = float(os.environ.get("YOLO_IOU", "0.50"))
YOLO_IMGSZ = int(os.environ.get("YOLO_IMGSZ", "1280"))
YOLO_MIN_AREA = int(os.environ.get("YOLO_MIN_AREA", "60"))
YOLO_DEVICE = os.environ.get("YOLO_DEVICE", "auto")

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}

VIDEO_UPLOAD_DIR = BASE_DIR / "static" / "videos"
VIDEO_RESULT_DIR = BASE_DIR / "static" / "video_results"

VIDEO_MAX_SIZE = int(os.environ.get("VIDEO_MAX_SIZE", "500"))  # MB
VIDEO_TARGET_FPS = int(os.environ.get("VIDEO_TARGET_FPS", "0"))  # 0 = original fps
VIDEO_MAX_WIDTH = int(os.environ.get("VIDEO_MAX_WIDTH", "1280"))
VIDEO_PROGRESS_INTERVAL = int(os.environ.get("VIDEO_PROGRESS_INTERVAL", "5"))  # frames between progress updates
VIDEO_BATCH_SIZE = int(os.environ.get("VIDEO_BATCH_SIZE", "4"))  # batch size for GPU inference
VIDEO_USE_TRACKER = os.environ.get("VIDEO_USE_TRACKER", "1") == "1"  # enable BYTETracker
VIDEO_TRACKER_PERSIST = os.environ.get("VIDEO_TRACKER_PERSIST", "1") == "1"  # persist tracks across frames

API_PROVIDERS = {
    "openai": {
        "name": "OpenAI",
        "api_url": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-4o-mini",
    },
    "qwen": {
        "name": "\u901a\u4e49\u5343\u95ee\uff08\u767e\u70bc-\u5317\u4eac\uff09",
        "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "model": "qwen3-vl-plus",
    },
    "qwen_intl": {
        "name": "\u901a\u4e49\u5343\u95ee\uff08\u767e\u70bc-\u65b0\u52a0\u5761\uff09",
        "api_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
        "model": "qwen3-vl-plus",
    },
    "qwen_us": {
        "name": "\u901a\u4e49\u5343\u95ee\uff08\u767e\u70bc-\u7f8e\u56fd\uff09",
        "api_url": "https://dashscope-us.aliyuncs.com/compatible-mode/v1/chat/completions",
        "model": "qwen3-vl-plus",
    },
    "kimi": {
        "name": "Kimi\uff08\u6708\u4e4b\u6697\u9762\uff09",
        "api_url": "https://api.moonshot.cn/v1/chat/completions",
        "model": "kimi-k2.5",
    },
    "custom": {
        "name": "\u81ea\u5b9a\u4e49 OpenAI \u517c\u5bb9\u63a5\u53e3",
        "api_url": LLM_API_URL,
        "model": LLM_MODEL,
    },
}
