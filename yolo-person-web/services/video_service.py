import json
import threading
import time
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
                "unique_count": 0,
                "sum_count": 0,
                "detection_target": "person",
                "message": "",
                "eta_seconds": 0.0,
                "processing_fps": 0.0,
            },
        )
        current.update(updates)


def get_video_progress(video_id: int) -> dict:
    with _progress_lock:
        defaults = {
            "status": "unknown",
            "progress": 0,
            "current_frame": 0,
            "total_frames": 0,
            "current_person_count": 0,
            "current_count": 0,
            "total_persons": 0,
            "total_count": 0,
            "unique_count": 0,
            "sum_count": 0,
            "detection_target": "person",
            "message": "",
            "eta_seconds": 0.0,
            "processing_fps": 0.0,
        }
        current = _progress.get(video_id, {})
        result = dict(defaults)
        result.update(current)
        return result


def get_video_count_metadata_path(video_path: Path) -> Path:
    return video_path.with_suffix(".counts.json")


def save_video_count_metadata(
    video_path: Path,
    fps: float,
    person_counts: list[int],
    detection_target: str = "person",
    unique_count: int = 0,
    sum_count: int = 0,
) -> None:
    get_video_count_metadata_path(video_path).write_text(
        json.dumps(
            {
                "version": 2,
                "fps": fps,
                "detection_target": detection_target,
                "person_counts": person_counts,
                "counts": person_counts,
                "unique_count": unique_count,
                "sum_count": sum_count,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def load_video_count_metadata(video_path: Path) -> dict:
    metadata_path = get_video_count_metadata_path(video_path)
    if not metadata_path.exists():
        return {"fps": 0, "person_counts": [], "detection_target": "person", "unique_count": 0, "sum_count": 0}
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"fps": 0, "person_counts": [], "detection_target": "person", "unique_count": 0, "sum_count": 0}
    person_counts = metadata.get("person_counts") or metadata.get("counts")
    if not isinstance(person_counts, list):
        person_counts = []
    return {
        "fps": float(metadata.get("fps") or 0),
        "person_counts": [int(count or 0) for count in person_counts],
        "detection_target": metadata.get("detection_target") or "person",
        "unique_count": int(metadata.get("unique_count") or 0),
        "sum_count": int(metadata.get("sum_count") or 0),
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


def _iou(box_a: tuple, box_b: tuple) -> float:
    x1, y1, x2, y2 = box_a
    x1b, y1b, x2b, y2b = box_b
    xi1 = max(x1, x1b)
    yi1 = max(y1, y1b)
    xi2 = min(x2, x2b)
    yi2 = min(y2, y2b)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = (x2 - x1) * (y2 - y1) + (x2b - x1b) * (y2b - y1b) - inter
    return inter / union if union > 0 else 0.0


def _box_center_distance(box_a: tuple, box_b: tuple) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    acx = (ax1 + ax2) / 2
    acy = (ay1 + ay2) / 2
    bcx = (bx1 + bx2) / 2
    bcy = (by1 + by2) / 2
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5


def _box_diagonal(box: tuple) -> float:
    x1, y1, x2, y2 = box
    return max(1.0, ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)


def _resolve_canonical_track_id(
    detection: dict,
    frame_index: int,
    track_aliases: dict[int, int],
    track_memory: dict[int, dict],
    max_gap: int,
) -> int | None:
    raw_track_id = detection.get("track_id")
    if raw_track_id is None:
        return None
    if raw_track_id in track_aliases:
        canonical_id = track_aliases[raw_track_id]
    else:
        canonical_id = raw_track_id
        current_box = detection["box"]
        current_class_id = detection.get("class_id")
        best_match_id = None
        best_match_score = 0.0

        for known_id, known_track in track_memory.items():
            if known_id == raw_track_id:
                continue
            gap = frame_index - known_track["last_frame"]
            if gap <= 0 or gap > max_gap:
                continue
            if current_class_id is not None and known_track.get("class_id") != current_class_id:
                continue

            previous_box = known_track["box"]
            overlap = _iou(current_box, previous_box)
            center_distance = _box_center_distance(current_box, previous_box)
            distance_limit = max(_box_diagonal(current_box), _box_diagonal(previous_box)) * 0.65
            if overlap >= 0.18 or center_distance <= distance_limit:
                score = overlap + max(0.0, 1.0 - center_distance / distance_limit)
                if score > best_match_score:
                    best_match_score = score
                    best_match_id = known_id

        if best_match_id is not None:
            canonical_id = best_match_id
        track_aliases[raw_track_id] = canonical_id

    track_memory[canonical_id] = {
        "box": detection["box"],
        "class_id": detection.get("class_id"),
        "last_frame": frame_index,
    }
    return canonical_id


def _filter_duplicate_drawn_tracks(tracks: list[dict]) -> list[dict]:
    current_tracks = [track for track in tracks if track.get("missed", 0) == 0]
    filtered = list(current_tracks)

    for track in tracks:
        if track.get("missed", 0) == 0:
            continue
        duplicate = False
        for current in current_tracks:
            if track.get("class_id") != current.get("class_id"):
                continue
            overlap = _iou(track["box"], current["box"])
            center_distance = _box_center_distance(track["box"], current["box"])
            distance_limit = max(_box_diagonal(track["box"]), _box_diagonal(current["box"])) * 0.45
            if overlap >= 0.12 or center_distance <= distance_limit:
                duplicate = True
                break
        if not duplicate:
            filtered.append(track)

    return filtered


def _smooth_detections(
    raw_detections: list[dict],
    smooth_tracks: list[dict],
    iou_threshold: float = 0.5,
    max_miss: int = 2,
) -> tuple[list[dict], list[dict]]:
    matched_track = set()
    matched_det = set()
    for di, det in enumerate(raw_detections):
        best_iou = 0.0
        best_ti = -1
        for ti, track in enumerate(smooth_tracks):
            if ti in matched_track:
                continue
            val = _iou(det["box"], track["box"])
            if val > best_iou:
                best_iou = val
                best_ti = ti
        if best_iou >= iou_threshold:
            matched_track.add(best_ti)
            matched_det.add(di)
            smooth_tracks[best_ti]["box"] = det["box"]
            smooth_tracks[best_ti]["conf"] = det["conf"]
            smooth_tracks[best_ti]["class_id"] = det.get("class_id")
            smooth_tracks[best_ti]["class_name"] = det.get("class_name", "person")
            smooth_tracks[best_ti]["missed"] = 0

    for ti, track in enumerate(smooth_tracks):
        if ti not in matched_track:
            track["missed"] += 1

    for di, det in enumerate(raw_detections):
        if di not in matched_det:
            smooth_tracks.append({**det, "missed": 0})

    active_tracks = [t for t in smooth_tracks if t["missed"] <= max_miss]
    drawn_tracks = _filter_duplicate_drawn_tracks(active_tracks)

    drawn_detections = [
        {
            "box": t["box"],
            "conf": t["conf"],
            "class_id": t.get("class_id"),
            "class_name": t.get("class_name", "person"),
        }
        for t in drawn_tracks
    ]
    return drawn_detections, active_tracks


def _extract_detections(result) -> list[dict]:
    detections = []
    if result.boxes is None:
        return detections

    xyxy_list = result.boxes.xyxy.cpu().tolist()
    conf_list = result.boxes.conf.cpu().tolist()
    cls_list = result.boxes.cls.cpu().tolist() if result.boxes.cls is not None else [None] * len(xyxy_list)
    names = getattr(result, "names", {}) or {}
    track_ids = None
    if result.boxes.id is not None:
        track_ids = result.boxes.id.int().tolist()

    for i, (xyxy, confidence, class_id) in enumerate(zip(xyxy_list, conf_list, cls_list)):
        x1, y1, x2, y2 = [int(round(value)) for value in xyxy]
        if x2 <= x1 or y2 <= y1:
            continue
        class_id = int(class_id) if class_id is not None else None
        class_name = VEHICLE_CLASS_LABELS.get(class_id) or names.get(class_id) or "person"
        det = {
            "box": (x1, y1, x2, y2),
            "conf": float(confidence),
            "class_id": class_id,
            "class_name": class_name,
        }
        if track_ids is not None:
            det["track_id"] = int(track_ids[i])
        detections.append(det)
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
    model_name = getattr(record, "yolo_model_name", "") or None
    model = get_yolo_model(model_name)
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
        model_name=model_name,
        message="Processing video...",
    )

    writer = None
    processed = 0
    total_persons = 0
    confidence_sum = 0.0
    confidence_count = 0
    frame_person_counts = []
    last_db_update = time.time()
    DB_UPDATE_INTERVAL = 1.0  # 最小数据库写入间隔（秒）
    smooth_tracks = []
    track_aliases = {}
    track_memory = {}
    unique_track_ids = set()
    SMOOTH_MAX_MISS = 2
    SMOOTH_IOU_THRESHOLD = 0.5

    start_time = time.time()
    try:
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or record.fps or 25)
        fps = VIDEO_TARGET_FPS if VIDEO_TARGET_FPS > 0 else source_fps
        track_merge_max_gap = max(8, int(source_fps * 1.5))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or record.video_width)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or record.video_height)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise ValueError("Cannot create output video file")

        batch_size = 1 if VIDEO_USE_TRACKER else max(1, VIDEO_BATCH_SIZE)

        while True:
            frames = []
            for _ in range(batch_size):
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
                "max_det": 1000,
                "verbose": False,
            }
            device = resolve_yolo_device()
            if device:
                predict_kwargs["device"] = device

            results = model.track(persist=True, **predict_kwargs) if VIDEO_USE_TRACKER else model.predict(**predict_kwargs)
            for frame, result in zip(frames, results):
                raw_detections = _extract_detections(result)
                drawn_detections, smooth_tracks = _smooth_detections(
                    raw_detections,
                    smooth_tracks,
                    SMOOTH_IOU_THRESHOLD,
                    SMOOTH_MAX_MISS,
                )
                current_person_count = len(drawn_detections)
                frame_person_counts.append(current_person_count)
                _draw_detections(frame, drawn_detections, cv2, target_options["box_color"])
                writer.write(frame)
                processed += 1
                total_persons = max(total_persons, current_person_count)
                for det in raw_detections:
                    confidence_sum += det["conf"]
                    confidence_count += 1
                    canonical_track_id = _resolve_canonical_track_id(
                        det,
                        processed,
                        track_aliases,
                        track_memory,
                        track_merge_max_gap,
                    )
                    if canonical_track_id is not None:
                        unique_track_ids.add(canonical_track_id)

                if processed % max(1, VIDEO_PROGRESS_INTERVAL) == 0:
                    now = time.time()
                    if now - last_db_update >= DB_UPDATE_INTERVAL:
                        last_db_update = now
                        progress = int(processed / record.total_frames * 100) if record.total_frames else 0
                        record.processed_frames = processed
                        db.session.commit()

                        elapsed = now - start_time
                        processing_fps = processed / elapsed if elapsed > 0 else 0.0
                        eta_seconds = (record.total_frames - processed) / processing_fps if processing_fps > 0 and record.total_frames else 0.0

                        update_video_progress(
                            video_id,
                            status="processing",
                            progress=min(progress, 99),
                            current_frame=processed,
                            total_frames=record.total_frames or 0,
                            current_person_count=current_person_count,
                            current_count=current_person_count,
                            total_persons=total_persons,
                            total_count=len(unique_track_ids) or total_persons,
                            unique_count=len(unique_track_ids),
                            sum_count=sum(frame_person_counts),
                            detection_target=detection_target,
                            eta_seconds=round(eta_seconds, 1),
                            processing_fps=round(processing_fps, 1),
                            message=f"Processed {processed}/{record.total_frames or '?'} frames",
                        )

        avg_confidence = confidence_sum / confidence_count if confidence_count else 0.0
        sum_count = sum(frame_person_counts)
        record.status = "completed"
        record.processed_video_path = str(output_path)
        record.processed_frames = processed
        record.total_persons = total_persons
        record.unique_count = len(unique_track_ids)
        record.sum_count = sum_count
        record.avg_confidence = avg_confidence
        save_video_count_metadata(output_path, fps, frame_person_counts, detection_target, len(unique_track_ids), sum_count)
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
            total_count=len(unique_track_ids) or total_persons,
            unique_count=len(unique_track_ids),
            sum_count=sum_count,
            detection_target=detection_target,
            eta_seconds=0.0,
            processing_fps=0.0,
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
        # 处理完成后清理内存中的进度缓存，防止长期运行内存泄漏
        with _progress_lock:
            _progress.pop(video_id, None)
