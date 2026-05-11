import json
import threading
from pathlib import Path
from uuid import uuid4

from config import (
    ALLOWED_VIDEO_EXTENSIONS,
    VIDEO_BATCH_SIZE,
    VIDEO_PROGRESS_INTERVAL,
    VIDEO_RESULT_DIR,
    VIDEO_TARGET_FPS,
    VIDEO_USE_TRACKER,
    YOLO_CONF,
    YOLO_IMGSZ,
    YOLO_IOU,
)
from models import VideoRecord, db
from services.yolo_service import (
    DEFAULT_PERSON_CLASS_IDS,
    VEHICLE_CLASS_IDS,
    VEHICLE_CLASS_LABELS,
    get_yolo_model,
    resolve_yolo_device,
)

_progress = {}
_progress_lock = threading.Lock()

VIDEO_DETECTION_TARGETS = {
    "person": {
        "class_ids": DEFAULT_PERSON_CLASS_IDS,
        "box_color": (22, 163, 74),
    },
    "vehicle": {
        "class_ids": VEHICLE_CLASS_IDS,
        "box_color": (15, 118, 110),
    },
}


def get_video_detection_target(target: str | None) -> dict:
    return VIDEO_DETECTION_TARGETS.get(target or "person", VIDEO_DETECTION_TARGETS["person"])


def allowed_video_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_VIDEO_EXTENSIONS


def update_video_progress(video_id: int, **updates) -> None:
    with _progress_lock:
        current = _progress.setdefault(
            video_id,
            {
                "status": "unknown",
                "progress": 0,
                "current_frame": 0,
                "total_frames": 0,
                "current_person_count": 0,
                "current_count": 0,
                "total_persons": 0,
                "total_count": 0,
                "detection_target": "person",
                "message": "",
            },
        )
        current.update(updates)


def get_video_progress(video_id: int) -> dict:
    with _progress_lock:
        return dict(
            _progress.get(
                video_id,
                {
                    "status": "unknown",
                    "progress": 0,
                    "current_frame": 0,
                    "total_frames": 0,
                    "current_person_count": 0,
                    "current_count": 0,
                    "total_persons": 0,
                    "total_count": 0,
                    "detection_target": "person",
                    "message": "",
                },
            )
        )


def get_video_count_metadata_path(video_path: Path) -> Path:
    return video_path.with_suffix(".counts.json")


def save_video_count_metadata(
    video_path: Path,
    fps: float,
    person_counts: list[int],
    detection_target: str = "person",
) -> None:
    get_video_count_metadata_path(video_path).write_text(
        json.dumps(
            {
                "version": 2,
                "fps": fps,
                "detection_target": detection_target,
                "person_counts": person_counts,
                "counts": person_counts,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def load_video_count_metadata(video_path: Path) -> dict:
    metadata_path = get_video_count_metadata_path(video_path)
    if not metadata_path.exists():
        return {"fps": 0, "person_counts": [], "detection_target": "person"}
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"fps": 0, "person_counts": [], "detection_target": "person"}
    person_counts = metadata.get("person_counts") or metadata.get("counts")
    if not isinstance(person_counts, list):
        person_counts = []
    return {
        "fps": float(metadata.get("fps") or 0),
        "person_counts": [int(count or 0) for count in person_counts],
        "detection_target": metadata.get("detection_target") or "person",
    }


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Missing opencv-python. Install it before processing video.") from exc
    return cv2


def get_video_info(video_path: Path) -> dict:
    cv2 = _require_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Cannot open video file")
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = total_frames / fps if fps > 0 else 0
        return {"total_frames": total_frames, "fps": fps, "duration": duration, "width": width, "height": height}
    finally:
        cap.release()


def _extract_detections(result) -> list[dict]:
    detections = []
    if result.boxes is None:
        return detections

    xyxy_list = result.boxes.xyxy.cpu().tolist()
    conf_list = result.boxes.conf.cpu().tolist()
    cls_list = result.boxes.cls.cpu().tolist() if result.boxes.cls is not None else [None] * len(xyxy_list)
    names = getattr(result, "names", {}) or {}

    for xyxy, confidence, class_id in zip(xyxy_list, conf_list, cls_list):
        x1, y1, x2, y2 = [int(round(value)) for value in xyxy]
        if x2 <= x1 or y2 <= y1:
            continue
        class_id = int(class_id) if class_id is not None else None
        class_name = VEHICLE_CLASS_LABELS.get(class_id) or names.get(class_id) or "person"
        detections.append(
            {
                "box": (x1, y1, x2, y2),
                "conf": float(confidence),
                "class_id": class_id,
                "class_name": class_name,
            }
        )
    return detections


def _draw_detections(frame, detections, cv2, box_color) -> None:
    for index, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = det["box"]
        conf = det["conf"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        label = f"{det.get('class_name') or index} {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_w, label_h = label_size
        top = max(0, y1 - label_h - 8)
        cv2.rectangle(frame, (x1, top), (x1 + label_w + 10, top + label_h + 8), box_color, -1)
        cv2.putText(frame, label, (x1 + 5, top + label_h + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def process_video_detection(video_id: int) -> None:
    cv2 = _require_cv2()
    record = db.session.get(VideoRecord, video_id)
    if record is None:
        update_video_progress(video_id, status="failed", message="Video record not found")
        return

    video_path = Path(record.video_path)
    output_path = VIDEO_RESULT_DIR / f"{uuid4().hex}.mp4"
    model = get_yolo_model()
    detection_target = getattr(record, "detection_target", "person") or "person"
    target_options = get_video_detection_target(detection_target)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Cannot open video file")

    record.status = "processing"
    db.session.commit()
    update_video_progress(
        video_id,
        status="processing",
        progress=0,
        current_frame=0,
        total_frames=record.total_frames or 0,
        detection_target=detection_target,
        message="Processing video...",
    )

    writer = None
    processed = 0
    total_persons = 0
    confidence_sum = 0.0
    confidence_count = 0
    frame_person_counts = []

    try:
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or record.fps or 25)
        fps = VIDEO_TARGET_FPS if VIDEO_TARGET_FPS > 0 else source_fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or record.video_width)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or record.video_height)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise ValueError("Cannot create output video file")

        while True:
            frames = []
            for _ in range(max(1, VIDEO_BATCH_SIZE)):
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
            if not frames:
                break

            predict_kwargs = {
                "source": frames,
                "classes": target_options["class_ids"],
                "conf": YOLO_CONF,
                "iou": YOLO_IOU,
                "imgsz": YOLO_IMGSZ,
                "max_det": 300,
                "verbose": False,
            }
            device = resolve_yolo_device()
            if device:
                predict_kwargs["device"] = device

            results = model.track(persist=True, **predict_kwargs) if VIDEO_USE_TRACKER else model.predict(**predict_kwargs)
            for frame, result in zip(frames, results):
                detections = _extract_detections(result)
                current_person_count = len(detections)
                frame_person_counts.append(current_person_count)
                _draw_detections(frame, detections, cv2, target_options["box_color"])
                writer.write(frame)
                processed += 1
                total_persons = max(total_persons, current_person_count)
                for det in detections:
                    confidence_sum += det["conf"]
                    confidence_count += 1

                if processed % max(1, VIDEO_PROGRESS_INTERVAL) == 0:
                    progress = int(processed / record.total_frames * 100) if record.total_frames else 0
                    record.processed_frames = processed
                    db.session.commit()
                    update_video_progress(
                        video_id,
                        status="processing",
                        progress=min(progress, 99),
                        current_frame=processed,
                        total_frames=record.total_frames or 0,
                        current_person_count=current_person_count,
                        current_count=current_person_count,
                        total_persons=total_persons,
                        total_count=total_persons,
                        detection_target=detection_target,
                        message=f"Processed {processed}/{record.total_frames or '?'} frames",
                    )

        avg_confidence = confidence_sum / confidence_count if confidence_count else 0.0
        record.status = "completed"
        record.processed_video_path = str(output_path)
        record.processed_frames = processed
        record.total_persons = total_persons
        record.avg_confidence = avg_confidence
        save_video_count_metadata(output_path, fps, frame_person_counts, detection_target)
        db.session.commit()
        final_count = frame_person_counts[-1] if frame_person_counts else 0
        update_video_progress(
            video_id,
            status="completed",
            progress=100,
            current_frame=processed,
            total_frames=record.total_frames or processed,
            current_person_count=final_count,
            current_count=final_count,
            total_persons=total_persons,
            total_count=total_persons,
            detection_target=detection_target,
            message="Processing complete",
        )
    except Exception as exc:
        record.status = "failed"
        record.error_message = str(exc)
        db.session.commit()
        update_video_progress(video_id, status="failed", message=str(exc), detection_target=detection_target)
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        raise
    finally:
        cap.release()
        if writer is not None:
            writer.release()
