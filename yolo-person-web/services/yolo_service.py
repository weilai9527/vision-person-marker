import json
from pathlib import Path

from PIL import Image, ImageOps
from ultralytics import YOLO

from config import (
    AUTO_LLM_COMPLEX_PERSONS,
    AUTO_LLM_LOW_CONF,
    AUTO_LLM_SMALL_BOX_RATIO,
    YOLO_CONF,
    YOLO_DEVICE,
    YOLO_IMGSZ,
    YOLO_IOU,
    YOLO_MIN_AREA,
    YOLO_MODEL_PATH,
)
from models import DetectionResult, db
from services.image_service import draw_person_boxes as default_draw_person_boxes

_yolo_model = None
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


def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        if not YOLO_MODEL_PATH.exists():
            raise ValueError(f"YOLO \u6a21\u578b\u6587\u4ef6\u4e0d\u5b58\u5728\uff1a{YOLO_MODEL_PATH}")
        _yolo_model = YOLO(str(YOLO_MODEL_PATH))
    return _yolo_model


def run_yolo_detection(
    image_path: Path,
    class_ids: list[int] | None = None,
    target_label: str = "person",
    conf: float | None = None,
    iou: float | None = None,
    imgsz: int | None = None,
    min_area: int | None = None,
) -> tuple[list[dict], int, int]:
    model = get_yolo_model()
    class_ids = class_ids or DEFAULT_PERSON_CLASS_IDS

    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        predict_kwargs = {
            "source": image,
            "classes": class_ids,
            "conf": YOLO_CONF if conf is None else conf,
            "iou": YOLO_IOU if iou is None else iou,
            "imgsz": YOLO_IMGSZ if imgsz is None else imgsz,
            "max_det": 300,
            "verbose": False,
        }
        device = resolve_yolo_device()
        if device:
            predict_kwargs["device"] = device
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


def build_yolo_analysis(target_label: str, count_label: str, drawn_count: int, prefix: str = "") -> str:
    return (
        prefix
        + f"\u4f7f\u7528\u672c\u5730 YOLO \u68c0\u6d4b\uff0c\u7edf\u8ba1 {target_label} \u7c7b\u522b\u3002"
        f"\u6a21\u578b\uff1a{YOLO_MODEL_PATH.name}\uff0cconf={YOLO_CONF}\uff0ciou={YOLO_IOU}\uff0cimgsz={YOLO_IMGSZ}\u3002"
        f"\u5171\u7ed8\u5236 {drawn_count} \u4e2a{count_label}\u6846\u3002"
    )


def is_complex_person_scene(boxes: list[dict], width: int, height: int) -> tuple[bool, str]:
    count = len(boxes)
    if count >= AUTO_LLM_COMPLEX_PERSONS:
        return True, f"YOLO \u9884\u5224\u4eba\u6570 {count} \u8f83\u591a"

    if not boxes or width <= 0 or height <= 0:
        return False, "\u672a\u68c0\u6d4b\u5230\u590d\u6742\u4eba\u7fa4"

    confidences = [float(box.get("conf", 1)) for box in boxes]
    low_conf_count = sum(1 for confidence in confidences if confidence < AUTO_LLM_LOW_CONF)
    if count >= 4 and low_conf_count >= max(2, count // 3):
        return True, f"YOLO \u6709 {low_conf_count} \u4e2a\u4f4e\u7f6e\u4fe1\u5ea6\u5019\u9009\u6846"

    image_area = width * height
    small_boxes = 0
    for box in boxes:
        box_area = max(0, box["x2"] - box["x1"]) * max(0, box["y2"] - box["y1"])
        if box_area / image_area < AUTO_LLM_SMALL_BOX_RATIO:
            small_boxes += 1
    if count >= 6 and small_boxes >= max(4, count // 2):
        return True, f"YOLO \u9884\u5224\u5c0f\u76ee\u6807\u8f83\u591a\uff08{small_boxes}/{count}\uff09"

    return False, "\u573a\u666f\u8f83\u7b80\u5355"


def save_detection_result(
    image_record_id: int,
    person_count: int,
    boxes: list[dict],
    analysis: str,
    result_name: str,
    api_config: dict,
) -> None:
    detection_result = DetectionResult(
        image_id=image_record_id,
        person_count=person_count,
        bounding_boxes_json=json.dumps(boxes, ensure_ascii=False),
        llm_analysis_text=analysis,
        result_image_path=f"static/results/{result_name}",
        llm_api_provider=api_config.get("provider", "local_yolo"),
        llm_model_name=YOLO_MODEL_PATH.name,
        raw_llm_response_log_path=None,
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
    count_label: str = "\u884c\u4eba",
    analysis_prefix: str = "",
    precomputed_boxes: list[dict] | None = None,
    precomputed_size: tuple[int, int] | None = None,
) -> tuple[int, str, str, int, int]:
    if precomputed_boxes is not None and precomputed_size is not None:
        boxes = precomputed_boxes
        width, height = precomputed_size
    else:
        boxes, width, height = run_yolo_detection(image_path, class_ids, target_label)
    result_name, drawn_count = draw_person_boxes(image_path, boxes, (width, height))
    analysis = build_yolo_analysis(target_label, count_label, drawn_count, analysis_prefix)
    save_detection_result(image_record_id, drawn_count, boxes, analysis, result_name, api_config)

    return drawn_count, analysis, result_name, width, height
