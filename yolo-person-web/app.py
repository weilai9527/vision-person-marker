import base64
from datetime import datetime
from io import BytesIO
import json
import mimetypes
import os
import re
from pathlib import Path
from uuid import uuid4

from flask import Flask, render_template, request, url_for
import requests
from PIL import Image, ImageDraw, ImageOps
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
RESULT_DIR = BASE_DIR / "static" / "results"
LOG_DIR = BASE_DIR / "logs"
API_CONFIG_PATH = BASE_DIR / "api_config.json"
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
YOLO_DEVICE = os.environ.get("YOLO_DEVICE", "")

_yolo_model = None

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
API_PROVIDERS = {
    "openai": {
        "name": "OpenAI",
        "api_url": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-4o-mini",
    },
    "qwen": {
        "name": "通义千问（百炼-北京）",
        "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "model": "qwen3-vl-plus",
    },
    "qwen_intl": {
        "name": "通义千问（百炼-新加坡）",
        "api_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
        "model": "qwen3-vl-plus",
    },
    "qwen_us": {
        "name": "通义千问（百炼-美国）",
        "api_url": "https://dashscope-us.aliyuncs.com/compatible-mode/v1/chat/completions",
        "model": "qwen3-vl-plus",
    },
    "kimi": {
        "name": "Kimi（月之暗面）",
        "api_url": "https://api.moonshot.cn/v1/chat/completions",
        "model": "kimi-k2.5",
    },
    "custom": {
        "name": "自定义 OpenAI 兼容接口",
        "api_url": LLM_API_URL,
        "model": LLM_MODEL,
    },
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def image_to_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    image_base64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{image_base64}"


def choose_model_image_size(width: int, height: int) -> tuple[int, int, int]:
    if MODEL_IMAGE_MAX_SIDE > 0:
        max_side = MODEL_IMAGE_MAX_SIDE
    else:
        total_pixels = width * height
        if total_pixels > 12_000_000:
            max_side = 1200
        elif total_pixels > 6_000_000:
            max_side = 1400
        elif total_pixels > 2_500_000:
            max_side = 1600
        else:
            max_side = 1800

    scale_by_side = min(1.0, max_side / max(width, height))
    scale_by_pixels = min(1.0, (MODEL_IMAGE_MAX_PIXELS / (width * height)) ** 0.5)
    scale = min(scale_by_side, scale_by_pixels)

    if scale < 1.0:
        target_width = max(1, round(width * scale))
        target_height = max(1, round(height * scale))
        if min(target_width, target_height) < MODEL_IMAGE_MIN_SIDE and min(width, height) >= MODEL_IMAGE_MIN_SIDE:
            min_side_scale = MODEL_IMAGE_MIN_SIDE / min(width, height)
            if min_side_scale < scale_by_side:
                target_width = max(1, round(width * min_side_scale))
                target_height = max(1, round(height * min_side_scale))
    else:
        target_width, target_height = width, height

    return target_width, target_height, max_side


def pil_image_to_data_url(image: Image.Image) -> tuple[str, int]:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=MODEL_IMAGE_JPEG_QUALITY, optimize=True)
    image_base64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{image_base64}", buffer.tell()


def prepare_model_image_object(image_path: Path) -> tuple[Image.Image, dict]:
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        original_width, original_height = image.size
        target_width, target_height, max_side = choose_model_image_size(original_width, original_height)
        if (target_width, target_height) != image.size:
            image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        return image.copy(), {
            "original_width": original_width,
            "original_height": original_height,
            "model_width": image.width,
            "model_height": image.height,
            "adaptive_max_side": max_side,
        }


def prepare_model_image(image_path: Path) -> tuple[str, int, int, dict]:
    image, image_info = prepare_model_image_object(image_path)
    data_url, jpeg_bytes = pil_image_to_data_url(image)
    image_info = {
        **image_info,
        "jpeg_bytes": jpeg_bytes,
    }
    return data_url, image.width, image.height, image_info


def parse_model_json(content: str) -> dict:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)

    candidates = [
        cleaned,
        repair_json_text(cleaned),
    ]

    last_error = None
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc

    fallback = extract_detection_from_text(cleaned)
    if fallback is not None:
        return fallback

    preview = cleaned[:1000]
    raise ValueError(f"模型没有返回严格 JSON，解析失败：{last_error}。返回内容片段：{preview}")


def save_model_response_log(
    image_path: Path,
    api_config: dict,
    response_data: dict,
    content: str,
    image_info: dict,
) -> str:
    log_name = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}.json"
    log_path = LOG_DIR / log_name
    log_payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "provider": api_config.get("provider"),
        "api_url": api_config.get("api_url"),
        "model": api_config.get("model"),
        "image": image_path.name,
        "image_info": image_info,
        "raw_content": content,
        "raw_response": response_data,
    }
    log_path.write_text(json.dumps(log_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return log_name


def repair_json_text(text: str) -> str:
    repaired = text.strip()
    repaired = repaired.replace("，", ",").replace("：", ":")
    repaired = repaired.replace("“", '"').replace("”", '"').replace("‘", '"').replace("’", '"')
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    repaired = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', repaired)
    return repaired


def extract_detection_from_text(text: str) -> dict | None:
    count_match = re.search(r'"?person_count"?\s*[:：]\s*(\d+)', text)
    if not count_match:
        count_match = re.search(r"(?:人数|检测到|person_count)[^\d]{0,12}(\d+)", text)

    boxes = []
    first_malformed_box = re.search(
        r'"?x1"?\s*[:：]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]',
        text,
    )
    if first_malformed_box:
        boxes.append(
            {
                "x1": float(first_malformed_box.group(1)),
                "y1": float(first_malformed_box.group(2)),
                "x2": float(first_malformed_box.group(3)),
                "y2": float(first_malformed_box.group(4)),
            }
        )

    for match in re.finditer(
        r'"?x\d+"?\s*[:：]\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]',
        text,
    ):
        boxes.append(
            {
                "x1": float(match.group(1)),
                "y1": float(match.group(2)),
                "x2": float(match.group(3)),
                "y2": float(match.group(4)),
            }
        )

    for match in re.finditer(
        r'"?x1"?\s*[:：]\s*(-?\d+(?:\.\d+)?).*?'
        r'"?y1"?\s*[:：]\s*(-?\d+(?:\.\d+)?).*?'
        r'"?x2"?\s*[:：]\s*(-?\d+(?:\.\d+)?).*?'
        r'"?y2"?\s*[:：]\s*(-?\d+(?:\.\d+)?)',
        text,
        flags=re.DOTALL,
    ):
        boxes.append(
            {
                "x1": float(match.group(1)),
                "y1": float(match.group(2)),
                "x2": float(match.group(3)),
                "y2": float(match.group(4)),
            }
        )

    if not count_match and not boxes:
        return None

    person_count = int(count_match.group(1)) if count_match else len(boxes)
    return {
        "person_count": person_count,
        "boxes": boxes,
        "analysis": "模型返回的 JSON 不完整，系统已尽量提取人数和坐标。",
    }


def normalize_chat_endpoint(api_url: str) -> str:
    cleaned = api_url.strip().rstrip("/")
    if not cleaned:
        return API_PROVIDERS["openai"]["api_url"]
    if cleaned.endswith("/chat/completions"):
        return cleaned
    if cleaned.endswith("/v1"):
        return f"{cleaned}/chat/completions"
    return cleaned


def infer_provider(api_url: str) -> str:
    if "dashscope-intl" in api_url:
        return "qwen_intl"
    if "dashscope-us" in api_url:
        return "qwen_us"
    if "dashscope" in api_url:
        return "qwen"
    if "moonshot" in api_url:
        return "kimi"
    if "api.openai.com" in api_url:
        return "openai"
    return "custom"


def normalize_model_name(provider: str, model: str) -> str:
    cleaned = model.strip()
    if provider.startswith("qwen"):
        return cleaned.lower()
    return cleaned


def first_present(data: dict, keys: tuple[str, ...]):
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def normalize_box(box: dict | list, width: int, height: int) -> tuple[int, int, int, int] | None:
    raw_values = extract_box_values(box)
    if raw_values is None:
        return None

    try:
        x1, y1, x2, y2 = [float(value) for value in raw_values]
    except (TypeError, ValueError):
        return None

    return normalize_raw_box((x1, y1, x2, y2), width, height)


def extract_box_values(box: dict | list) -> list | None:
    if isinstance(box, dict):
        nested_box = (
            box.get("bbox")
            or box.get("box")
            or box.get("box_2d")
            or box.get("bbox_2d")
            or box.get("coordinates")
            or box.get("position")
        )
        if isinstance(nested_box, list) and len(nested_box) >= 4:
            return nested_box[:4]
        elif isinstance(box.get("x1"), list) and len(box["x1"]) >= 4:
            return box["x1"][:4]
        elif isinstance(nested_box, dict):
            return [
                first_present(nested_box, ("x1", "left")),
                first_present(nested_box, ("y1", "top")),
                first_present(nested_box, ("x2", "right")),
                first_present(nested_box, ("y2", "bottom")),
            ]
        else:
            return [
                first_present(box, ("x1", "left")),
                first_present(box, ("y1", "top")),
                first_present(box, ("x2", "right")),
                first_present(box, ("y2", "bottom")),
            ]
    elif isinstance(box, list) and len(box) >= 4:
        return box[:4]
    return None


def normalize_raw_box(raw_box: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = raw_box
    try:
        max_value = max(x1, y1, x2, y2)
    except (TypeError, ValueError):
        return None

    if max_value <= 1.0:
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    elif max_value <= 1000 and (x2 > width or y2 > height):
        x1, x2 = x1 / 1000 * width, x2 / 1000 * width
        y1, y2 = y1 / 1000 * height, y2 / 1000 * height

    left = max(0, min(width - 1, round(min(x1, x2))))
    top = max(0, min(height - 1, round(min(y1, y2))))
    right = max(0, min(width - 1, round(max(x1, x2))))
    bottom = max(0, min(height - 1, round(max(y1, y2))))

    if right - left < 4 or bottom - top < 4:
        return None
    return left, top, right, bottom


def infer_box_coordinate_size(boxes: list, default_width: int, default_height: int) -> tuple[int, int]:
    raw_boxes = []
    for box in boxes:
        raw_values = extract_box_values(box)
        if raw_values is None:
            continue
        try:
            raw_boxes.append(tuple(float(value) for value in raw_values[:4]))
        except (TypeError, ValueError):
            continue

    if not raw_boxes:
        return default_width, default_height

    max_x = max(max(box[0], box[2]) for box in raw_boxes)
    max_y = max(max(box[1], box[3]) for box in raw_boxes)
    max_value = max(max_x, max_y)

    if max_value <= 1.0:
        return 1, 1
    if max_value <= 1000 and (default_width > 1200 or default_height > 1200):
        return 1000, 1000
    return default_width, default_height


def expand_box(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
    expand_x: float = BOX_EXPAND_X,
    expand_y: float = BOX_EXPAND_Y,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    box_width = right - left
    box_height = bottom - top
    pad_x = round(box_width * expand_x)
    pad_y = round(box_height * expand_y)
    return (
        max(0, left - pad_x),
        max(0, top - pad_y),
        min(width - 1, right + pad_x),
        min(height - 1, bottom + pad_y),
    )


def box_iou(first: dict, second: dict) -> float:
    left = max(first["x1"], second["x1"])
    top = max(first["y1"], second["y1"])
    right = min(first["x2"], second["x2"])
    bottom = min(first["y2"], second["y2"])
    if right <= left or bottom <= top:
        return 0.0

    intersection = (right - left) * (bottom - top)
    first_area = (first["x2"] - first["x1"]) * (first["y2"] - first["y1"])
    second_area = (second["x2"] - second["x1"]) * (second["y2"] - second["y1"])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def merge_boxes(boxes: list[dict]) -> list[dict]:
    valid_boxes = [
        box for box in boxes
        if box["x2"] > box["x1"] and box["y2"] > box["y1"]
    ]
    valid_boxes.sort(key=lambda box: (box["x2"] - box["x1"]) * (box["y2"] - box["y1"]), reverse=True)

    merged = []
    for box in valid_boxes:
        if all(box_iou(box, kept_box) < NMS_IOU_THRESHOLD for kept_box in merged):
            merged.append(box)
    return merged


def limit_boxes(boxes: list[dict], max_count: int) -> list[dict]:
    if len(boxes) <= max_count:
        return boxes
    return boxes[:max_count]


def convert_boxes_to_model_coords(
    boxes: list,
    source_width: int,
    source_height: int,
    offset_x: int = 0,
    offset_y: int = 0,
) -> list[dict]:
    coord_width, coord_height = infer_box_coordinate_size(boxes, source_width, source_height)
    scale_x = source_width / coord_width
    scale_y = source_height / coord_height
    converted = []

    for box in boxes:
        normalized_box = normalize_box(box, coord_width, coord_height)
        if normalized_box is None:
            continue
        left, top, right, bottom = normalized_box
        converted.append(
            {
                "x1": round(left * scale_x + offset_x),
                "y1": round(top * scale_y + offset_y),
                "x2": round(right * scale_x + offset_x),
                "y2": round(bottom * scale_y + offset_y),
            }
        )
    return converted


def generate_tiles(width: int, height: int) -> list[tuple[int, int, int, int]]:
    if TILE_GRID <= 1:
        return []

    step_x = width / TILE_GRID
    step_y = height / TILE_GRID
    overlap_x = round(step_x * TILE_OVERLAP)
    overlap_y = round(step_y * TILE_OVERLAP)
    tiles = []

    for row in range(TILE_GRID):
        for col in range(TILE_GRID):
            left = max(0, round(col * step_x) - overlap_x)
            top = max(0, round(row * step_y) - overlap_y)
            right = min(width, round((col + 1) * step_x) + overlap_x)
            bottom = min(height, round((row + 1) * step_y) + overlap_y)
            if right - left >= 120 and bottom - top >= 120:
                tiles.append((left, top, right, bottom))
    return tiles


def draw_person_boxes(image_path: Path, boxes: list, box_image_size: tuple[int, int] | None = None) -> tuple[str, int]:
    result_name = f"{uuid4().hex}{image_path.suffix.lower()}"
    result_path = RESULT_DIR / result_name

    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        default_box_width, default_box_height = box_image_size or (width, height)
        box_width, box_height = infer_box_coordinate_size(boxes, default_box_width, default_box_height)
        scale_x = width / box_width
        scale_y = height / box_height
        draw = ImageDraw.Draw(image)
        line_width = max(3, min(width, height) // 160)
        drawn_count = 0

        for index, box in enumerate(boxes, start=1):
            normalized_box = normalize_box(box, box_width, box_height)
            if normalized_box is None:
                continue

            left, top, right, bottom = normalized_box
            left = max(0, min(width - 1, round(left * scale_x)))
            top = max(0, min(height - 1, round(top * scale_y)))
            right = max(0, min(width - 1, round(right * scale_x)))
            bottom = max(0, min(height - 1, round(bottom * scale_y)))
            left, top, right, bottom = expand_box((left, top, right, bottom), width, height)
            draw.rectangle((left, top, right, bottom), outline="#16a34a", width=line_width)
            drawn_count += 1
            label = str(index)
            text_box = draw.textbbox((0, 0), label)
            label_width = text_box[2] - text_box[0] + 12
            label_height = text_box[3] - text_box[1] + 8
            label_top = max(0, top - label_height)
            draw.rectangle(
                (left, label_top, left + label_width, label_top + label_height),
                fill="#16a34a",
            )
            draw.text((left + 6, label_top + 4), label, fill="#ffffff")

        image.save(result_path)

    return result_name, drawn_count


def load_api_config() -> dict:
    config = {
        "provider": "openai",
        "api_url": LLM_API_URL,
        "api_key": LLM_API_KEY,
        "model": LLM_MODEL,
    }
    if API_CONFIG_PATH.exists():
        try:
            saved_config = json.loads(API_CONFIG_PATH.read_text(encoding="utf-8"))
            config.update({key: saved_config.get(key) or value for key, value in config.items()})
        except (OSError, json.JSONDecodeError):
            pass
    config["api_url"] = normalize_chat_endpoint(config["api_url"])
    if config["provider"] not in API_PROVIDERS or config["provider"] == "openai":
        config["provider"] = infer_provider(config["api_url"])
    config["model"] = normalize_model_name(config["provider"], config["model"])
    return config


def save_api_config(provider: str, api_url: str, api_key: str, model: str) -> None:
    provider = provider if provider in API_PROVIDERS else "custom"
    provider_defaults = API_PROVIDERS[provider]
    normalized_model = normalize_model_name(provider, model) or provider_defaults["model"]
    config = {
        "provider": provider,
        "api_url": normalize_chat_endpoint(api_url or provider_defaults["api_url"]),
        "api_key": api_key.strip(),
        "model": normalized_model,
    }
    API_CONFIG_PATH.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        if not YOLO_MODEL_PATH.exists():
            raise ValueError(f"YOLO 模型文件不存在：{YOLO_MODEL_PATH}")
        _yolo_model = YOLO(str(YOLO_MODEL_PATH))
    return _yolo_model


def call_local_yolo(image_path: Path) -> tuple[int, str, str]:
    model = get_yolo_model()

    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        predict_kwargs = {
            "source": image,
            "classes": [0],
            "conf": YOLO_CONF,
            "iou": YOLO_IOU,
            "imgsz": YOLO_IMGSZ,
            "max_det": 300,
            "verbose": False,
        }
        if YOLO_DEVICE:
            predict_kwargs["device"] = YOLO_DEVICE
        results = model.predict(**predict_kwargs)

    boxes = []
    result = results[0]
    if result.boxes is not None:
        xyxy_list = result.boxes.xyxy.cpu().tolist()
        conf_list = result.boxes.conf.cpu().tolist()

        for xyxy, confidence in zip(xyxy_list, conf_list):
            x1, y1, x2, y2 = [int(round(value)) for value in xyxy]
            if x2 <= x1 or y2 <= y1:
                continue

            area = (x2 - x1) * (y2 - y1)
            if area < YOLO_MIN_AREA:
                continue

            boxes.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "conf": round(float(confidence), 3),
                }
            )

    result_name, drawn_count = draw_person_boxes(image_path, boxes, (width, height))
    analysis = (
        "使用本地 YOLO 检测，只统计 person 类别。"
        f"模型：{YOLO_MODEL_PATH.name}，conf={YOLO_CONF}，iou={YOLO_IOU}，imgsz={YOLO_IMGSZ}。"
        f"共绘制 {drawn_count} 个行人框。"
    )
    return drawn_count, analysis, result_name


def build_detection_prompt(width: int, height: int, scope: str = "整张图片") -> str:
    prompt = (
        f"请检测{scope}中可见的人数，并给出每个真实人物的外接矩形框。图片尺寸为 {width}x{height} 像素。"
        "只统计真实可见的人，不统计照片、海报、雕像或屏幕里的人。"
        "每一个计入 person_count 的人都必须在 boxes 中有且只有一个外接矩形框，boxes 数量必须等于 person_count，不能多也不能少。"
        "不要把同一个人拆成多个框，不要为车辆、影子、反光、标牌或身体部位单独建框。"
        "不要漏掉远处、小尺寸、遮挡或边缘位置的真实人物。"
        "坐标必须使用当前图片像素坐标，左上角为 (0,0)，字段为 x1,y1,x2,y2。"
        "必须只返回一行紧凑 JSON，不要 Markdown，不要解释，不要换行，不要尾逗号。格式为："
        '{"person_count": 0, "boxes": [{"x1": 0, "y1": 0, "x2": 0, "y2": 0}], "analysis": "简短中文说明"}'
    )
    return prompt


def request_detection(
    api_config: dict,
    image_url: str,
    width: int,
    height: int,
    scope: str,
) -> tuple[dict, dict]:
    payload = {
        "model": api_config["model"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_detection_prompt(width, height, scope)},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "temperature": 0,
    }
    headers = {
        "Authorization": f"Bearer {api_config['api_key']}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(api_config["api_url"], headers=headers, json=payload, timeout=LLM_TIMEOUT)
    except requests.exceptions.Timeout as exc:
        raise ValueError(
            f"接口响应超时：已等待 {LLM_TIMEOUT} 秒。人多或图片复杂时千问视觉模型会更慢，"
            "请重试，或调大环境变量 LLM_TIMEOUT。"
        ) from exc
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = response.text.strip()
        if len(detail) > 800:
            detail = f"{detail[:800]}..."
        raise ValueError(f"接口请求失败：HTTP {response.status_code}，地址：{api_config['api_url']}，响应：{detail}") from exc

    data = response.json()
    content = data["choices"][0]["message"]["content"]
    result = parse_model_json(content)
    return result, {"response": data, "content": content}


def call_vision_model(image_path: Path) -> tuple[int, str, str]:
    api_config = load_api_config()

    if not api_config["api_key"]:
        raise ValueError("未配置 LLM_API_KEY，请先设置大模型 API Key。")

    model_image, model_image_info = prepare_model_image_object(image_path)
    model_image_url, jpeg_bytes = pil_image_to_data_url(model_image)
    model_image_info = {**model_image_info, "jpeg_bytes": jpeg_bytes}
    model_image_width, model_image_height = model_image.size

    result, raw_log = request_detection(
        api_config,
        model_image_url,
        model_image_width,
        model_image_height,
        "整张图片",
    )
    log_payload = {
        "global": raw_log,
        "tiles": [],
    }
    log_name = save_model_response_log(image_path, api_config, log_payload, raw_log["content"], model_image_info)

    try:
        person_count = int(result["person_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"模型返回结果缺少有效 person_count：{content}") from exc

    analysis = str(result.get("analysis") or "模型未返回说明。")
    boxes = result.get("boxes") or result.get("persons") or result.get("people") or result.get("detections") or []
    if not isinstance(boxes, list):
        boxes = []
    if len(boxes) > person_count:
        boxes = boxes[:person_count]
    merged_boxes = convert_boxes_to_model_coords(boxes, model_image_width, model_image_height)

    tile_added = 0
    tile_errors = 0
    if ENABLE_TILE_DETECTION:
        for tile_index, (left, top, right, bottom) in enumerate(generate_tiles(model_image_width, model_image_height), start=1):
            crop = model_image.crop((left, top, right, bottom))
            tile_url, _ = pil_image_to_data_url(crop)
            try:
                tile_result, tile_log = request_detection(
                    api_config,
                    tile_url,
                    crop.width,
                    crop.height,
                    f"第 {tile_index} 个局部区域",
                )
            except Exception as exc:
                tile_errors += 1
                log_payload["tiles"].append(
                    {
                        "tile": [left, top, right, bottom],
                        "error": str(exc),
                    }
                )
                continue

            tile_boxes = (
                tile_result.get("boxes")
                or tile_result.get("persons")
                or tile_result.get("people")
                or tile_result.get("detections")
                or []
            )
            if isinstance(tile_boxes, list) and len(merged_boxes) < person_count:
                before_count = len(merged_boxes)
                merged_boxes = merge_boxes(
                    merged_boxes
                    + convert_boxes_to_model_coords(tile_boxes, crop.width, crop.height, left, top)
                )
                merged_boxes = limit_boxes(merged_boxes, person_count)
                tile_added += max(0, len(merged_boxes) - before_count)
            log_payload["tiles"].append(
                {
                    "tile": [left, top, right, bottom],
                    "result": tile_result,
                    "raw_content": tile_log["content"],
                    "raw_response": tile_log["response"],
                }
            )

        log_path = LOG_DIR / log_name
        if log_path.exists():
            saved_log = json.loads(log_path.read_text(encoding="utf-8"))
            saved_log["raw_response"] = log_payload
            log_path.write_text(json.dumps(saved_log, ensure_ascii=False, indent=2), encoding="utf-8")
        if tile_added:
            analysis = f"{analysis}（分块补检新增 {tile_added} 个候选框。）"
        if tile_errors:
            analysis = f"{analysis}（有 {tile_errors} 个分块补检失败。）"

    result_name, drawn_count = draw_person_boxes(image_path, merged_boxes, (model_image_width, model_image_height))
    if drawn_count != person_count:
        analysis = (
            f"{analysis}（注意：模型识别人数为 {person_count}，"
            f"合并后成功绘制的框为 {drawn_count} 个。）"
        )
    analysis = f"{analysis}（原始返回已保存：logs/{log_name}）"
    return person_count, analysis, result_name


@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    success = None
    person_count = None
    analysis = None
    result_image_url = None
    original_filename = None
    api_config = load_api_config()

    if request.method == "POST":
        action = request.form.get("action")
        if action == "save_api":
            provider = request.form.get("provider", "custom")
            api_url = request.form.get("api_url", "")
            api_key = request.form.get("api_key", "")
            model = request.form.get("model", "")
            if not api_key.strip():
                error = "请填写 API Key。"
            else:
                save_api_config(provider, api_url, api_key, model)
                api_config = load_api_config()
                success = "API 配置已保存。"
        else:
            file = request.files.get("image")
            if file is None or file.filename == "":
                error = "请先选择一张图片。"
            elif not allowed_file(file.filename):
                error = "只支持 jpg、jpeg、png、bmp、webp 格式。"
            else:
                suffix = Path(file.filename).suffix.lower()
                upload_name = f"{uuid4().hex}{suffix}"
                upload_path = UPLOAD_DIR / upload_name
                file.save(upload_path)

                try:
                    person_count, analysis, result_name = call_local_yolo(upload_path)
                    result_image_url = url_for("static", filename=f"results/{result_name}")
                    original_filename = file.filename
                except Exception as exc:
                    error = f"检测失败：{exc}"

    return render_template(
        "index.html",
        error=error,
        success=success,
        api_config=api_config,
        api_providers=API_PROVIDERS,
        has_api_key=bool(api_config.get("api_key")),
        person_count=person_count,
        analysis=analysis,
        result_image_url=result_image_url,
        original_filename=original_filename,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
