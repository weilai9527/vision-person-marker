import json
import threading
from pathlib import Path

from PIL import Image, ImageOps

from config import (
    AUTO_LLM_COMPLEX_PERSONS,
    AUTO_LLM_LOW_CONF,
    AUTO_LLM_SMALL_BOX_RATIO,
    YOLO_CONF,
    YOLO_DEVICE,
    YOLO_HALF,
    YOLO_IMGSZ,
    YOLO_IOU,
    YOLO_MIN_AREA,
    YOLO_MODEL_PATH,
)
from models import DetectionResult, db
from services.image_service import draw_person_boxes as default_draw_person_boxes

_yolo_models = {}
_yolo_model_lock = threading.Lock()
_inference_lock = threading.Lock()
_resolved_device = None

DEFAULT_PERSON_CLASS_IDS = [0]
VEHICLE_CLASS_IDS = [1, 2, 3, 5, 7]
VEHICLE_CLASS_LABELS = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def resolve_yolo_device() -> str | None:
    global _resolved_device
    if _resolved_device is not None:
        return _resolved_device

    configured = (YOLO_DEVICE or "").strip().lower()
    if configured and configured not in {"auto", "none", "cpu"}:
        _resolved_device = YOLO_DEVICE
        return _resolved_device
    if configured == "cpu" or configured == "none":
        _resolved_device = None
        return _resolved_device

    try:
        import torch
    except ImportError:
        _resolved_device = None
        return _resolved_device

    _resolved_device = "0" if torch.cuda.is_available() else None
    return _resolved_device


def should_use_half_precision(device: str | None) -> bool:
    configured = (YOLO_HALF or "").strip().lower()
    if configured in {"1", "true", "yes", "on"}:
        return True
    if configured in {"0", "false", "no", "off", "cpu"}:
        return False
    return bool(device)


def get_yolo_model_dir() -> Path:
    return YOLO_MODEL_PATH.parent


def get_default_yolo_model_name() -> str:
    return YOLO_MODEL_PATH.name


def normalize_yolo_model_name(model_name: str | None = None) -> str:
    selected = (model_name or "").strip() or get_default_yolo_model_name()
    candidate = Path(selected)
    if candidate.name != selected or candidate.suffix.lower() != ".pt":
        raise ValueError("YOLO 模型名称不合法")

    model_path = get_yolo_model_dir() / selected
    if not model_path.exists():
        raise ValueError(f"YOLO 模型文件不存在：{selected}")
    return selected


def get_available_yolo_models() -> list[dict]:
    model_dir = get_yolo_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(model_dir.glob("*.pt"), key=lambda path: path.name.lower())
    default_name = get_default_yolo_model_name()
    models = []
    for path in files:
        name = path.name
        lower_name = name.lower()
        if lower_name.endswith(("n.pt", "nano.pt")):
            tier = "极速"
        elif lower_name.endswith(("s.pt", "small.pt")):
            tier = "推荐"
        elif lower_name.endswith(("m.pt", "medium.pt")):
            tier = "均衡"
        elif lower_name in {"best.pt", "custom.pt"}:
            tier = "自训练"
        else:
            tier = "模型"
        models.append(
            {
                "name": name,
                "label": name,
                "size_mb": round(path.stat().st_size / (1024 * 1024), 1),
                "current": name == default_name,
            }
        )
    return models


def get_yolo_model(model_name: str | None = None):
    selected_name = normalize_yolo_model_name(model_name)
    model_path = get_yolo_model_dir() / selected_name
    cache_key = str(model_path.resolve())
    model = _yolo_models.get(cache_key)
    if model is None:
        with _yolo_model_lock:
            model = _yolo_models.get(cache_key)
            if model is None:
                from ultralytics import YOLO

                model = YOLO(str(model_path))
                device = resolve_yolo_device()
                if device:
                    model.to(f"cuda:{device}" if device.isdigit() else device)
                _yolo_models[cache_key] = model
    return model


def run_yolo_detection(
    image_path: Path,
    class_ids: list[int] | None = None,
    target_label: str = "person",
    conf: float | None = None,
    iou: float | None = None,
    imgsz: int | None = None,
    min_area: int | None = None,
    max_det: int = 1000,
    model_name: str | None = None,
) -> tuple[list[dict], int, int]:
    with Image.open(image_path) as image:
        return run_yolo_detection_on_image(
            image,
            class_ids=class_ids,
            target_label=target_label,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            min_area=min_area,
            max_det=max_det,
            model_name=model_name,
        )


def run_yolo_detection_on_image(
    image: Image.Image,
    class_ids: list[int] | None = None,
    target_label: str = "person",
    conf: float | None = None,
    iou: float | None = None,
    imgsz: int | None = None,
    min_area: int | None = None,
    max_det: int = 1000,
    model_name: str | None = None,
) -> tuple[list[dict], int, int]:
    model = get_yolo_model(model_name)
    class_ids = DEFAULT_PERSON_CLASS_IDS if class_ids is None else class_ids
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    predict_kwargs = {
        "source": image,
        "classes": class_ids,
        "conf": YOLO_CONF if conf is None else conf,
        "iou": YOLO_IOU if iou is None else iou,
        "imgsz": YOLO_IMGSZ if imgsz is None else imgsz,
        "max_det": max_det,
        "verbose": False,
    }
    device = resolve_yolo_device()
    if device:
        predict_kwargs["device"] = device
    if should_use_half_precision(device):
        predict_kwargs["half"] = True
    with _inference_lock:
        results = model.predict(**predict_kwargs)

    boxes = []
    result = results[0]
    if result.boxes is not None:
        xyxy_list = result.boxes.xyxy.cpu().tolist()
        conf_list = result.boxes.conf.cpu().tolist()
        cls_list = result.boxes.cls.cpu().tolist() if result.boxes.cls is not None else [None] * len(xyxy_list)

        for xyxy, confidence, class_id in zip(xyxy_list, conf_list, cls_list):
            x1, y1, x2, y2 = [int(round(value)) for value in xyxy]
            if x2 <= x1 or y2 <= y1:
                continue
            area = (x2 - x1) * (y2 - y1)
            if area < (YOLO_MIN_AREA if min_area is None else min_area):
                continue
            class_id = int(class_id) if class_id is not None else None
            class_name = VEHICLE_CLASS_LABELS.get(class_id) or getattr(model, "names", {}).get(class_id, target_label)
            boxes.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "conf": round(float(confidence), 3),
                    "class_id": class_id,
                    "class_name": class_name,
                    "label": f"{class_name} {len(boxes) + 1}",
                }
            )

    return boxes, width, height


def reset_yolo_tracker(model_name: str | None = None) -> None:
    model = get_yolo_model(model_name)
    predictor = getattr(model, "predictor", None)
    trackers = getattr(predictor, "trackers", None)
    if not trackers:
        return
    for tracker in trackers:
        reset = getattr(tracker, "reset", None)
        if callable(reset):
            reset()


def run_yolo_tracking_on_image(
    image: Image.Image,
    class_ids: list[int] | None = None,
    target_label: str = "person",
    conf: float | None = None,
    iou: float | None = None,
    imgsz: int | None = None,
    min_area: int | None = None,
    max_det: int = 1000,
    persist: bool = True,
    model_name: str | None = None,
) -> tuple[list[dict], int, int]:
    model = get_yolo_model(model_name)
    class_ids = DEFAULT_PERSON_CLASS_IDS if class_ids is None else class_ids
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    track_kwargs = {
        "source": image,
        "classes": class_ids,
        "conf": YOLO_CONF if conf is None else conf,
        "iou": YOLO_IOU if iou is None else iou,
        "imgsz": YOLO_IMGSZ if imgsz is None else imgsz,
        "max_det": max_det,
        "persist": persist,
        "verbose": False,
    }
    device = resolve_yolo_device()
    if device:
        track_kwargs["device"] = device
    if should_use_half_precision(device):
        track_kwargs["half"] = True

    with _inference_lock:
        results = model.track(**track_kwargs)

    boxes = []
    result = results[0]
    if result.boxes is not None:
        xyxy_list = result.boxes.xyxy.cpu().tolist()
        conf_list = result.boxes.conf.cpu().tolist()
        cls_list = result.boxes.cls.cpu().tolist() if result.boxes.cls is not None else [None] * len(xyxy_list)
        track_ids = result.boxes.id.int().cpu().tolist() if result.boxes.id is not None else None

        for index, (xyxy, confidence, class_id) in enumerate(zip(xyxy_list, conf_list, cls_list)):
            x1, y1, x2, y2 = [int(round(value)) for value in xyxy]
            if x2 <= x1 or y2 <= y1:
                continue
            area = (x2 - x1) * (y2 - y1)
            if area < (YOLO_MIN_AREA if min_area is None else min_area):
                continue
            class_id = int(class_id) if class_id is not None else None
            class_name = VEHICLE_CLASS_LABELS.get(class_id) or getattr(model, "names", {}).get(class_id, target_label)
            box = {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "conf": round(float(confidence), 3),
                "class_id": class_id,
                "class_name": class_name,
                "label": f"{class_name} {len(boxes) + 1}",
            }
            if track_ids is not None:
                box["raw_track_id"] = int(track_ids[index])
            boxes.append(box)

    return boxes, width, height


def build_yolo_analysis(target_label: str, count_label: str, drawn_count: int, prefix: str = "") -> str:
    return (
        prefix
        + f"使用本地 YOLO 检测，统计 {target_label} 类别。"
        f"模型：{YOLO_MODEL_PATH.name}，conf={YOLO_CONF}，iou={YOLO_IOU}，imgsz={YOLO_IMGSZ}。"
        f"共绘制 {drawn_count} 个{count_label}框。"
    )


def is_complex_detection_scene(boxes: list[dict], width: int, height: int, target_label: str = "人") -> tuple[bool, str]:
    count = len(boxes)
    if count >= AUTO_LLM_COMPLEX_PERSONS:
        return True, f"YOLO 预判{target_label}数 {count} 较多"

    if not boxes or width <= 0 or height <= 0:
        return False, f"未检测到复杂{target_label}群"

    image_area = width * height
    low_conf_count = 0
    small_boxes = 0

    for box in boxes:
        if float(box.get("conf", 1)) < AUTO_LLM_LOW_CONF:
            low_conf_count += 1
        box_area = max(0, box["x2"] - box["x1"]) * max(0, box["y2"] - box["y1"])
        if box_area / image_area < AUTO_LLM_SMALL_BOX_RATIO:
            small_boxes += 1

    if count >= 4 and low_conf_count >= max(2, count // 3):
        return True, f"YOLO 有 {low_conf_count} 个低置信度候选框"

    if count >= 6 and small_boxes >= max(4, count // 2):
        return True, f"YOLO 预判小目标较多（{small_boxes}/{count}）"

    return False, "场景较简单"


def is_suspected_yolo_miss(
    boxes: list[dict],
    width: int,
    height: int,
    target_label: str = "目标",
    empty_threshold: int = 0,
    low_conf_threshold: float = 0.35,
    max_low_conf_count: int = 2,
) -> tuple[bool, str]:
    if width <= 0 or height <= 0:
        return False, "图片尺寸无效"

    if not boxes:
        return True, f"YOLO 未检测到任何{target_label}"

    if len(boxes) <= empty_threshold:
        return True, f"YOLO 仅检测到 {len(boxes)} 个{target_label}，数量过少"

    low_conf_count = sum(1 for box in boxes if float(box.get("conf", 1)) < low_conf_threshold)
    if len(boxes) <= max_low_conf_count and low_conf_count == len(boxes):
        return True, f"YOLO 仅检测到 {len(boxes)} 个低置信度{target_label}"

    return False, "未发现明显漏检风险"


# 保留兼容别名
def is_complex_person_scene(boxes: list[dict], width: int, height: int) -> tuple[bool, str]:
    return is_complex_detection_scene(boxes, width, height, target_label="人")


def save_detection_result(
    image_record_id: int,
    person_count: int,
    boxes: list[dict],
    analysis: str,
    result_name: str,
    api_config: dict,
    model_name: str | None = None,
    raw_yolo_boxes: list[dict] | None = None,
    llm_boxes: list[dict] | None = None,
    final_source: str | None = None,
    detection_strategy: str | None = None,
    yolo_miss_reason: str | None = None,
) -> None:
    detection_result = DetectionResult(
        image_id=image_record_id,
        person_count=person_count,
        bounding_boxes_json=json.dumps(boxes, ensure_ascii=False),
        llm_analysis_text=analysis,
        result_image_path=f"static/results/{result_name}",
        llm_api_provider=api_config.get("provider", "local_yolo"),
        llm_model_name=model_name or YOLO_MODEL_PATH.name,
        raw_llm_response_log_path=None,
        raw_yolo_boxes_json=json.dumps(raw_yolo_boxes, ensure_ascii=False) if raw_yolo_boxes else None,
        llm_boxes_json=json.dumps(llm_boxes, ensure_ascii=False) if llm_boxes else None,
        final_source=final_source or api_config.get("provider", "local_yolo"),
        review_status="pending",
        detection_strategy=detection_strategy,
        yolo_miss_reason=yolo_miss_reason,
    )
    db.session.add(detection_result)
    db.session.commit()


def call_local_yolo(
    image_path: Path,
    image_record_id: int,
    api_config: dict,
    draw_person_boxes=default_draw_person_boxes,
    class_ids: list[int] | None = None,
    target_label: str = "person",
    count_label: str = "行人",
    analysis_prefix: str = "",
    precomputed_boxes: list[dict] | None = None,
    precomputed_size: tuple[int, int] | None = None,
    model_name: str | None = None,
    raw_yolo_boxes: list[dict] | None = None,
    llm_boxes: list[dict] | None = None,
    final_source: str | None = None,
    detection_strategy: str | None = None,
    yolo_miss_reason: str | None = None,
) -> tuple[int, str, str, int, int]:
    if precomputed_boxes is not None and precomputed_size is not None:
        boxes = precomputed_boxes
        width, height = precomputed_size
    else:
        boxes, width, height = run_yolo_detection(image_path, class_ids, target_label, model_name=model_name)
    result_name, drawn_count = draw_person_boxes(image_path, boxes, (width, height))
    analysis = build_yolo_analysis(target_label, count_label, drawn_count, analysis_prefix)
    if model_name:
        analysis += f" 当前选择模型：{model_name}。"
    save_detection_result(
        image_record_id, drawn_count, boxes, analysis, result_name, api_config, model_name,
        raw_yolo_boxes=raw_yolo_boxes or boxes,
        llm_boxes=llm_boxes,
        final_source=final_source or api_config.get("provider", "local_yolo"),
        detection_strategy=detection_strategy or "yolo_only",
        yolo_miss_reason=yolo_miss_reason,
    )

    return drawn_count, analysis, result_name, width, height
