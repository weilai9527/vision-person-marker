import logging
import threading
import time

from PIL import Image

from config import (
    CAMERA_CONF,
    CAMERA_DUPLICATE_CONTAINMENT,
    CAMERA_DUPLICATE_IOU,
    CAMERA_IOU,
    CAMERA_IMGSZ,
    CAMERA_MAX_DET,
    CAMERA_MIN_AREA,
    CAMERA_TRACK_CONFIRM_FRAMES,
    CAMERA_TRACK_MAX_MISSES,
    CAMERA_TRACK_MERGE_DISTANCE,
    CAMERA_USE_TRACKER,
)
from services.yolo_service import (
    DEFAULT_PERSON_CLASS_IDS,
    VEHICLE_CLASS_IDS,
    normalize_yolo_model_name,
    reset_yolo_tracker,
    run_yolo_detection_on_image,
    run_yolo_tracking_on_image,
)

logger = logging.getLogger(__name__)


def camera_box_area(box: dict) -> int:
    return max(0, int(box.get("x2", 0)) - int(box.get("x1", 0))) * max(
        0,
        int(box.get("y2", 0)) - int(box.get("y1", 0)),
    )


def camera_box_intersection(first: dict, second: dict) -> int:
    left = max(int(first.get("x1", 0)), int(second.get("x1", 0)))
    top = max(int(first.get("y1", 0)), int(second.get("y1", 0)))
    right = min(int(first.get("x2", 0)), int(second.get("x2", 0)))
    bottom = min(int(first.get("y2", 0)), int(second.get("y2", 0)))
    return max(0, right - left) * max(0, bottom - top)


def camera_box_iou(first: dict, second: dict) -> float:
    intersection = camera_box_intersection(first, second)
    if intersection <= 0:
        return 0.0
    union = camera_box_area(first) + camera_box_area(second) - intersection
    return intersection / union if union > 0 else 0.0


def camera_box_center_distance(first: dict, second: dict) -> float:
    first_cx = (float(first.get("x1", 0)) + float(first.get("x2", 0))) / 2
    first_cy = (float(first.get("y1", 0)) + float(first.get("y2", 0))) / 2
    second_cx = (float(second.get("x1", 0)) + float(second.get("x2", 0))) / 2
    second_cy = (float(second.get("y1", 0)) + float(second.get("y2", 0))) / 2
    return ((first_cx - second_cx) ** 2 + (first_cy - second_cy) ** 2) ** 0.5


def camera_box_diagonal(box: dict) -> float:
    return max(
        1.0,
        (
            (float(box.get("x2", 0)) - float(box.get("x1", 0))) ** 2
            + (float(box.get("y2", 0)) - float(box.get("y1", 0))) ** 2
        )
        ** 0.5,
    )


def camera_boxes_are_duplicates(candidate: dict, kept_box: dict) -> bool:
    candidate_area = camera_box_area(candidate)
    kept_area = camera_box_area(kept_box)
    if candidate_area <= 0 or kept_area <= 0:
        return False

    intersection = camera_box_intersection(candidate, kept_box)
    if intersection <= 0:
        return False

    union = candidate_area + kept_area - intersection
    iou = intersection / union if union > 0 else 0
    containment = intersection / min(candidate_area, kept_area)
    if iou >= CAMERA_DUPLICATE_IOU or containment >= CAMERA_DUPLICATE_CONTAINMENT:
        return True

    candidate_cx = (int(candidate["x1"]) + int(candidate["x2"])) / 2
    candidate_cy = (int(candidate["y1"]) + int(candidate["y2"])) / 2
    kept_cx = (int(kept_box["x1"]) + int(kept_box["x2"])) / 2
    kept_cy = (int(kept_box["y1"]) + int(kept_box["y2"])) / 2
    center_distance = ((candidate_cx - kept_cx) ** 2 + (candidate_cy - kept_cy) ** 2) ** 0.5
    candidate_diag = max(1, ((int(candidate["x2"]) - int(candidate["x1"])) ** 2 + (int(candidate["y2"]) - int(candidate["y1"])) ** 2) ** 0.5)
    kept_diag = max(1, ((int(kept_box["x2"]) - int(kept_box["x1"])) ** 2 + (int(kept_box["y2"]) - int(kept_box["y1"])) ** 2) ** 0.5)
    return iou >= 0.18 and center_distance <= max(candidate_diag, kept_diag) * 0.25


def filter_camera_boxes(boxes: list[dict]) -> list[dict]:
    valid_boxes = [
        box
        for box in boxes
        if box.get("x2", 0) > box.get("x1", 0)
        and box.get("y2", 0) > box.get("y1", 0)
        and camera_box_area(box) >= CAMERA_MIN_AREA
    ]
    valid_boxes.sort(key=lambda box: float(box.get("conf", 0)), reverse=True)

    kept_boxes = []
    for box in valid_boxes:
        if any(camera_boxes_are_duplicates(box, kept_box) for kept_box in kept_boxes):
            continue
        kept_boxes.append(box)
    return kept_boxes


def run_camera_target_detection(image: Image.Image, target: str, model_name: str | None = None) -> tuple[list[dict], list[dict], int, int]:
    class_ids = VEHICLE_CLASS_IDS if target == "vehicle" else DEFAULT_PERSON_CLASS_IDS
    raw_boxes, width, height = run_yolo_detection_on_image(
        image,
        class_ids=class_ids,
        target_label=target,
        conf=CAMERA_CONF,
        iou=CAMERA_IOU,
        imgsz=CAMERA_IMGSZ,
        min_area=1,
        max_det=CAMERA_MAX_DET,
        model_name=model_name,
    )
    boxes = filter_camera_boxes(raw_boxes)
    if raw_boxes and not boxes:
        boxes = raw_boxes[:1]

    for box in raw_boxes:
        box["detection_target"] = target
    for box in boxes:
        box["detection_target"] = target
        box["class_name"] = box.get("class_name") or target
    return raw_boxes, boxes, width, height


def run_camera_combined_detection(image: Image.Image, model_name: str | None = None) -> tuple[list[dict], list[dict], int, int]:
    raw_boxes, width, height = run_yolo_detection_on_image(
        image,
        class_ids=DEFAULT_PERSON_CLASS_IDS + VEHICLE_CLASS_IDS,
        target_label="object",
        conf=CAMERA_CONF,
        iou=CAMERA_IOU,
        imgsz=CAMERA_IMGSZ,
        min_area=1,
        max_det=CAMERA_MAX_DET,
        model_name=model_name,
    )

    person_raw = []
    vehicle_raw = []
    for box in raw_boxes:
        class_id = int(box.get("class_id", -1))
        if class_id in DEFAULT_PERSON_CLASS_IDS:
            box["detection_target"] = "person"
            box["class_name"] = "person"
            person_raw.append(box)
        elif class_id in VEHICLE_CLASS_IDS:
            box["detection_target"] = "vehicle"
            vehicle_raw.append(box)

    person_boxes = filter_camera_boxes(person_raw)
    vehicle_boxes = filter_camera_boxes(vehicle_raw)
    if person_raw and not person_boxes:
        person_boxes = person_raw[:1]
    if vehicle_raw and not vehicle_boxes:
        vehicle_boxes = vehicle_raw[:1]

    boxes = person_boxes + vehicle_boxes
    return person_raw + vehicle_raw, boxes, width, height


def run_camera_target_tracking(image: Image.Image, target: str, model_name: str | None = None) -> tuple[list[dict], list[dict], int, int]:
    class_ids = VEHICLE_CLASS_IDS if target == "vehicle" else DEFAULT_PERSON_CLASS_IDS
    raw_boxes, width, height = run_yolo_tracking_on_image(
        image,
        class_ids=class_ids,
        target_label=target,
        conf=CAMERA_CONF,
        iou=CAMERA_IOU,
        imgsz=CAMERA_IMGSZ,
        min_area=1,
        max_det=CAMERA_MAX_DET,
        persist=True,
        model_name=model_name,
    )
    if not raw_boxes:
        raw_boxes, width, height = run_yolo_detection_on_image(
            image,
            class_ids=class_ids,
            target_label=target,
            conf=CAMERA_CONF,
            iou=CAMERA_IOU,
            imgsz=CAMERA_IMGSZ,
            min_area=1,
            max_det=CAMERA_MAX_DET,
            model_name=model_name,
        )
    for box in raw_boxes:
        box["detection_target"] = target
        box["class_name"] = box.get("class_name") or target
    boxes = filter_camera_boxes(raw_boxes)
    if raw_boxes and not boxes:
        boxes = raw_boxes[:1]
    return raw_boxes, boxes, width, height


def run_camera_combined_tracking(image: Image.Image, model_name: str | None = None) -> tuple[list[dict], list[dict], int, int]:
    raw_boxes, width, height = run_yolo_tracking_on_image(
        image,
        class_ids=DEFAULT_PERSON_CLASS_IDS + VEHICLE_CLASS_IDS,
        target_label="object",
        conf=CAMERA_CONF,
        iou=CAMERA_IOU,
        imgsz=CAMERA_IMGSZ,
        min_area=1,
        max_det=CAMERA_MAX_DET,
        persist=True,
        model_name=model_name,
    )
    if not raw_boxes:
        raw_boxes, width, height = run_yolo_detection_on_image(
            image,
            class_ids=DEFAULT_PERSON_CLASS_IDS + VEHICLE_CLASS_IDS,
            target_label="object",
            conf=CAMERA_CONF,
            iou=CAMERA_IOU,
            imgsz=CAMERA_IMGSZ,
            min_area=1,
            max_det=CAMERA_MAX_DET,
            model_name=model_name,
        )

    person_raw = []
    vehicle_raw = []
    for box in raw_boxes:
        class_id = int(box.get("class_id", -1))
        if class_id in DEFAULT_PERSON_CLASS_IDS:
            box["detection_target"] = "person"
            box["class_name"] = "person"
            person_raw.append(box)
        elif class_id in VEHICLE_CLASS_IDS:
            box["detection_target"] = "vehicle"
            vehicle_raw.append(box)

    person_boxes = filter_camera_boxes(person_raw)
    vehicle_boxes = filter_camera_boxes(vehicle_raw)
    if person_raw and not person_boxes:
        person_boxes = person_raw[:1]
    if vehicle_raw and not vehicle_boxes:
        vehicle_boxes = vehicle_raw[:1]
    return person_raw + vehicle_raw, person_boxes + vehicle_boxes, width, height


_camera_track_lock = threading.Lock()
_camera_track_states: dict[str, dict] = {}
_camera_track_last_cleanup = time.time()
CAMERA_TRACK_STATE_TTL = 1800


def get_camera_track_state(session_id: str, target: str, reset: bool = False) -> dict:
    global _camera_track_last_cleanup
    state_key = f"{session_id}:{target}"
    with _camera_track_lock:
        now = time.time()
        if now - _camera_track_last_cleanup > CAMERA_TRACK_STATE_TTL:
            stale_keys = [
                key for key, state in _camera_track_states.items()
                if now - state.get("last_active", now) > CAMERA_TRACK_STATE_TTL
            ]
            for key in stale_keys:
                _camera_track_states.pop(key, None)
            _camera_track_last_cleanup = now

        if reset or state_key not in _camera_track_states:
            _camera_track_states[state_key] = {
                "frame_index": 0,
                "next_id": 1,
                "raw_aliases": {},
                "memory": {},
                "hits": {},
                "seen_ids": set(),
                "last_active": now,
            }
        else:
            _camera_track_states[state_key]["last_active"] = now
        return _camera_track_states[state_key]


def find_camera_memory_match(state: dict, box: dict, raw_track_id: int | None) -> int | None:
    best_id = None
    best_score = 0.0
    current_class_id = box.get("class_id")
    for track_id, memory in state["memory"].items():
        if raw_track_id is not None and track_id == state["raw_aliases"].get(raw_track_id):
            continue
        if current_class_id is not None and memory.get("class_id") != current_class_id:
            continue
        gap = state["frame_index"] - memory.get("last_frame", 0)
        if gap <= 0 or gap > CAMERA_TRACK_MAX_MISSES:
            continue

        previous_box = memory["box"]
        overlap = camera_box_iou(box, previous_box)
        distance = camera_box_center_distance(box, previous_box)
        distance_limit = max(camera_box_diagonal(box), camera_box_diagonal(previous_box)) * CAMERA_TRACK_MERGE_DISTANCE
        if overlap >= 0.12 or distance <= distance_limit:
            score = overlap + max(0.0, 1.0 - distance / max(1.0, distance_limit))
            if score > best_score:
                best_score = score
                best_id = track_id
    return best_id


def apply_camera_tracking_state(boxes: list[dict], session_id: str, target: str, reset: bool = False) -> dict:
    state = get_camera_track_state(session_id, target, reset)
    state["frame_index"] += 1
    active_ids = set()
    new_confirmed = 0

    for box in boxes:
        raw_track_id = box.get("raw_track_id")
        canonical_id = None
        if raw_track_id is not None:
            raw_track_id = int(raw_track_id)
            canonical_id = state["raw_aliases"].get(raw_track_id)

        if canonical_id is None:
            canonical_id = find_camera_memory_match(state, box, raw_track_id)

        if canonical_id is None:
            canonical_id = state["next_id"]
            state["next_id"] += 1

        if raw_track_id is not None:
            state["raw_aliases"][raw_track_id] = canonical_id

        state["hits"][canonical_id] = state["hits"].get(canonical_id, 0) + 1
        state["memory"][canonical_id] = {
            "box": dict(box),
            "class_id": box.get("class_id"),
            "target": box.get("detection_target"),
            "last_frame": state["frame_index"],
        }
        active_ids.add(canonical_id)
        box["track_id"] = canonical_id
        box["track_hits"] = state["hits"][canonical_id]
        box["track_confirmed"] = state["hits"][canonical_id] >= CAMERA_TRACK_CONFIRM_FRAMES
        if box["track_confirmed"] and canonical_id not in state["seen_ids"]:
            state["seen_ids"].add(canonical_id)
            new_confirmed += 1

    stale_ids = [
        track_id
        for track_id, memory in state["memory"].items()
        if state["frame_index"] - memory.get("last_frame", 0) > CAMERA_TRACK_MAX_MISSES
    ]
    for track_id in stale_ids:
        state["memory"].pop(track_id, None)

    return {
        "active_ids": active_ids,
        "new_confirmed": new_confirmed,
        "total_unique_count": len(state["seen_ids"]),
        "confirmed_active_count": sum(1 for track_id in active_ids if track_id in state["seen_ids"]),
    }


def process_camera_frame_image(
    image: Image.Image,
    target: str,
    session_id: str,
    reset_tracking: bool = False,
    model_name: str | None = None,
) -> dict:
    started_at = time.perf_counter()
    tracker_used = False
    tracker_fallback = False
    selected_model_name = normalize_yolo_model_name(model_name)

    if CAMERA_USE_TRACKER:
        if reset_tracking:
            reset_yolo_tracker(selected_model_name)
        try:
            if target == "both":
                raw_boxes, boxes, width, height = run_camera_combined_tracking(image, selected_model_name)
            else:
                raw_boxes, boxes, width, height = run_camera_target_tracking(image, target, selected_model_name)
            tracker_used = True
        except Exception:
            tracker_fallback = True
            logger.exception("YOLO track failed, falling back to predict")
            if target == "both":
                raw_boxes, boxes, width, height = run_camera_combined_detection(image, selected_model_name)
            else:
                raw_boxes, boxes, width, height = run_camera_target_detection(image, target, selected_model_name)
    elif target == "both":
        raw_boxes, boxes, width, height = run_camera_combined_detection(image, selected_model_name)
    else:
        raw_boxes, boxes, width, height = run_camera_target_detection(image, target, selected_model_name)

    track_stats = apply_camera_tracking_state(boxes, session_id, target, reset_tracking)
    elapsed_ms = round((time.perf_counter() - started_at) * 1000)
    person_count = sum(1 for box in boxes if box.get("detection_target") == "person")
    vehicle_count = sum(1 for box in boxes if box.get("detection_target") == "vehicle")
    return {
        "success": True,
        "person_count": person_count,
        "vehicle_count": vehicle_count,
        "count": len(boxes),
        "raw_count": len(raw_boxes),
        "filtered_count": len(boxes),
        "new_track_count": track_stats["new_confirmed"],
        "total_unique_count": track_stats["total_unique_count"],
        "confirmed_active_count": track_stats["confirmed_active_count"],
        "tracker_used": tracker_used,
        "tracker_fallback": tracker_fallback,
        "detection_target": target,
        "boxes": boxes,
        "width": width,
        "height": height,
        "elapsed_ms": elapsed_ms,
        "transport": "webrtc",
        "model_name": selected_model_name,
    }
