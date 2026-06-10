import threading

from PIL import Image, ImageOps

from config import YOLO_DEVICE, YOLO_MODEL_PATH


PERSON_CLASS_IDS = [0]
VEHICLE_CLASS_IDS = [1, 2, 3, 5, 7]
VEHICLE_CLASS_LABELS = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

_model = None
_model_lock = threading.Lock()
_resolved_device = None


def resolve_device() -> str | None:
    global _resolved_device
    if _resolved_device is not None:
        return _resolved_device

    configured = (YOLO_DEVICE or "").strip().lower()
    if configured and configured not in {"auto", "none", "cpu"}:
        _resolved_device = YOLO_DEVICE
        return _resolved_device
    if configured in {"none", "cpu"}:
        _resolved_device = None
        return _resolved_device

    try:
        import torch
    except ImportError:
        _resolved_device = None
        return _resolved_device

    _resolved_device = "0" if torch.cuda.is_available() else None
    return _resolved_device


def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                if not YOLO_MODEL_PATH.exists():
                    raise ValueError(f"YOLO 模型文件不存在：{YOLO_MODEL_PATH}")
                from ultralytics import YOLO

                _model = YOLO(str(YOLO_MODEL_PATH))
                device = resolve_device()
                if device:
                    _model.to(f"cuda:{device}" if device.isdigit() else device)
    return _model


def run_detection_on_image(
    image: Image.Image,
    class_ids: list[int],
    target_label: str,
    conf: float,
    iou: float,
    imgsz: int,
    min_area: int,
) -> tuple[list[dict], int, int]:
    model = get_model()
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size

    predict_kwargs = {
        "source": image,
        "classes": class_ids,
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "max_det": 300,
        "verbose": False,
    }
    device = resolve_device()
    if device:
        predict_kwargs["device"] = device

    results = model.predict(**predict_kwargs)
    result = results[0]
    boxes = []
    if result.boxes is None:
        return boxes, width, height

    xyxy_list = result.boxes.xyxy.cpu().tolist()
    conf_list = result.boxes.conf.cpu().tolist()
    cls_list = result.boxes.cls.cpu().tolist() if result.boxes.cls is not None else [None] * len(xyxy_list)

    for xyxy, confidence, class_id in zip(xyxy_list, conf_list, cls_list):
        x1, y1, x2, y2 = [int(round(value)) for value in xyxy]
        if x2 <= x1 or y2 <= y1:
            continue
        area = (x2 - x1) * (y2 - y1)
        if area < min_area:
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
