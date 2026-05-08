import json
from pathlib import Path

from PIL import Image, ImageOps
from ultralytics import YOLO

from config import YOLO_CONF, YOLO_DEVICE, YOLO_IMGSZ, YOLO_IOU, YOLO_MIN_AREA, YOLO_MODEL_PATH
from models import DetectionResult, db
from services.image_service import draw_person_boxes as default_draw_person_boxes

_yolo_model = None
_resolved_device = None


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
            raise ValueError(f"YOLO 模型文件不存在：{YOLO_MODEL_PATH}")
        _yolo_model = YOLO(str(YOLO_MODEL_PATH))
    return _yolo_model


def call_local_yolo(
    image_path: Path,
    image_record_id: int,
    api_config: dict,
    draw_person_boxes=default_draw_person_boxes,
) -> tuple[int, str, str, int, int]:
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
        device = resolve_yolo_device()
        if device:
            predict_kwargs["device"] = device
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
    detection_result = DetectionResult(
        image_id=image_record_id,
        person_count=drawn_count,
        bounding_boxes_json=json.dumps(boxes, ensure_ascii=False),
        llm_analysis_text=analysis,
        result_image_path=f"static/results/{result_name}",
        llm_api_provider=api_config.get("provider", "local_yolo"),
        llm_model_name=YOLO_MODEL_PATH.name,
        raw_llm_response_log_path=None,
    )
    db.session.add(detection_result)
    db.session.commit()

    return drawn_count, analysis, result_name, width, height
