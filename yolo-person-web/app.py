import json
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

from flask import Flask, render_template, request, url_for, Response, jsonify, stream_with_context, send_file
from PIL import Image

from config import (
    BASE_DIR,
    DATABASE_DIR,
    DATABASE_PATH,
    UPLOAD_DIR,
    RESULT_DIR,
    LOG_DIR,
    API_CONFIG_PATH,
    LLM_API_URL,
    LLM_API_KEY,
    LLM_MODEL,
    ALLOWED_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
    VIDEO_UPLOAD_DIR,
    VIDEO_RESULT_DIR,
    VIDEO_MAX_SIZE,
    API_PROVIDERS,
)
from models import db, ImageRecord, VideoRecord
from services.image_service import draw_person_boxes, draw_vehicle_boxes
from services.llm_service import call_yolo_judge_model
from services.yolo_service import VEHICLE_CLASS_IDS, call_local_yolo, is_complex_person_scene, run_yolo_detection
from services.video_service import (
    process_video_detection,
    get_video_progress,
    update_video_progress,
    allowed_video_file,
    get_video_info,
    get_video_count_metadata_path,
    load_video_count_metadata,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = max(16 * 1024 * 1024, VIDEO_MAX_SIZE * 1024 * 1024)
DATABASE_DIR.mkdir(parents=True, exist_ok=True)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + str(DATABASE_PATH)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

with app.app_context():
    db.create_all()

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_RESULT_DIR.mkdir(parents=True, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def mask_api_key(key: str) -> str:
    if not key or len(key) < 8:
        return ""
    return key[:4] + "*" * (len(key) - 8) + key[-4:]


def is_masked_api_key(key: str) -> bool:
    return "*" in (key or "")


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
    if is_masked_api_key(config.get("api_key", "")):
        config["api_key"] = ""
    return config


def load_api_config_for_display() -> dict:
    config = load_api_config()
    if config.get("api_key"):
        config["api_key_masked"] = mask_api_key(config["api_key"])
        config["api_key"] = ""
    else:
        config["api_key_masked"] = ""
    return config


def save_api_config(provider: str, api_url: str, api_key: str, model: str) -> None:
    provider = provider if provider in API_PROVIDERS else "custom"
    provider_defaults = API_PROVIDERS[provider]
    normalized_model = normalize_model_name(provider, model) or provider_defaults["model"]
    current_config = load_api_config()
    cleaned_api_key = api_key.strip()
    if not cleaned_api_key or is_masked_api_key(cleaned_api_key):
        cleaned_api_key = current_config.get("api_key", "")
    config = {
        "provider": provider,
        "api_url": normalize_chat_endpoint(api_url or provider_defaults["api_url"]),
        "api_key": cleaned_api_key,
        "model": normalized_model,
    }
    API_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    API_CONFIG_PATH.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


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


@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    success = None
    person_count = None
    analysis = None
    result_image_url = None
    original_filename = None
    api_config = load_api_config_for_display()

    if request.method == "POST":
        action = request.form.get("action")
        if action == "save_api":
            provider = request.form.get("provider", "custom")
            api_url = request.form.get("api_url", "")
            api_key = request.form.get("api_key", "")
            model = request.form.get("model", "")
            current_api_config = load_api_config()
            submitted_api_key = api_key.strip()
            if (not submitted_api_key or is_masked_api_key(submitted_api_key)) and not current_api_config.get("api_key"):
                error = "\u8bf7\u586b\u5199\u5b8c\u6574 API Key\uff0c\u4e0d\u80fd\u4f7f\u7528\u5df2\u63a9\u7801\u7684 Key\u3002"
            else:
                save_api_config(provider, api_url, api_key, model)
                api_config = load_api_config_for_display()
                success = "API \u914d\u7f6e\u5df2\u4fdd\u5b58\u3002"
        else:
            file = request.files.get("image")
            if file is None or file.filename == "":
                error = "\u8bf7\u5148\u9009\u62e9\u4e00\u5f20\u56fe\u7247\u3002"
            elif not allowed_file(file.filename):
                error = "\u53ea\u652f\u6301 jpg\u3001jpeg\u3001png\u3001bmp\u3001webp \u683c\u5f0f\u3002"
            else:
                original_filename = file.filename
                suffix = Path(original_filename).suffix.lower()
                upload_name = f"{uuid4().hex}{suffix}"
                upload_path = UPLOAD_DIR / upload_name
                file.save(upload_path)

                with Image.open(upload_path) as img:
                    original_width, original_height = img.size

                image_record = ImageRecord(
                    original_filename=original_filename,
                    original_image_path=str(upload_path),
                    original_width=original_width,
                    original_height=original_height,
                )
                db.session.add(image_record)
                db.session.commit()

                try:
                    detection_config = load_api_config()
                    if detection_config.get("api_key"):
                        preflight_boxes, preflight_width, preflight_height = run_yolo_detection(upload_path)
                        is_complex, complexity_reason = is_complex_person_scene(
                            preflight_boxes, preflight_width, preflight_height
                        )
                        if is_complex:
                            try:
                                person_count, analysis, result_name, model_width, model_height = call_yolo_judge_model(
                                    upload_path,
                                    image_record.id,
                                    detection_config,
                                    draw_person_boxes,
                                    preflight_boxes,
                                    preflight_width,
                                    preflight_height,
                                    analysis_prefix=f"YOLO 快速预判：{complexity_reason}，已交给大模型裁判复核。",
                                )
                            except Exception as llm_exc:
                                person_count, analysis, result_name, model_width, model_height = call_local_yolo(
                                    upload_path,
                                    image_record.id,
                                    {"provider": "auto_yolo_llm_fallback"},
                                    draw_person_boxes,
                                    analysis_prefix=(
                                        f"YOLO 快速预判：{complexity_reason}，但大模型调用失败，已回退本地 YOLO。"
                                        f"大模型错误：{llm_exc}。"
                                    ),
                                    precomputed_boxes=preflight_boxes,
                                    precomputed_size=(preflight_width, preflight_height),
                                )
                        else:
                            person_count, analysis, result_name, model_width, model_height = call_local_yolo(
                                upload_path,
                                image_record.id,
                                {"provider": "auto_yolo_simple"},
                                draw_person_boxes,
                                analysis_prefix=f"YOLO 快速预判：{complexity_reason}，直接使用本地结果。",
                                precomputed_boxes=preflight_boxes,
                                precomputed_size=(preflight_width, preflight_height),
                            )
                    else:
                        person_count, analysis, result_name, model_width, model_height = call_local_yolo(
                            upload_path, image_record.id, {"provider": "local_yolo"}, draw_person_boxes
                        )
                    image_record.model_image_path = str(upload_path)
                    image_record.model_width = model_width
                    image_record.model_height = model_height
                    db.session.commit()
                    result_image_url = url_for("static", filename=f"results/{result_name}")
                except Exception as exc:
                    error = f"\u68c0\u6d4b\u5931\u8d25\uff1a{exc}"
                    logger.exception("\u68c0\u6d4b\u8fc7\u7a0b\u4e2d\u51fa\u9519")
                    db.session.rollback()

    return render_template(
        "index.html",
        error=error,
        success=success,
        api_config=api_config,
        api_providers=API_PROVIDERS,
        has_api_key=bool(load_api_config().get("api_key")),
        person_count=person_count,
        analysis=analysis,
        result_image_url=result_image_url,
        original_filename=original_filename,
    )


@app.route("/history")
def history():
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 10, type=int)
    per_page = min(per_page, 50)
    pagination = ImageRecord.query.order_by(ImageRecord.uploaded_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    return render_template("history.html", pagination=pagination, records=pagination.items)


@app.route("/camera")
def camera_page():
    return render_template("camera.html")


@app.route("/vehicle", methods=["GET", "POST"])
def vehicle_page():
    error = None
    vehicle_count = None
    analysis = None
    result_image_url = None
    original_filename = None

    if request.method == "POST":
        file = request.files.get("image")
        if file is None or file.filename == "":
            error = "\u8bf7\u5148\u9009\u62e9\u4e00\u5f20\u56fe\u7247\u3002"
        elif not allowed_file(file.filename):
            error = "\u53ea\u652f\u6301 jpg\u3001jpeg\u3001png\u3001bmp\u3001webp \u683c\u5f0f\u3002"
        else:
            original_filename = file.filename
            suffix = Path(original_filename).suffix.lower()
            upload_name = f"{uuid4().hex}{suffix}"
            upload_path = UPLOAD_DIR / upload_name
            file.save(upload_path)

            with Image.open(upload_path) as img:
                original_width, original_height = img.size

            image_record = ImageRecord(
                original_filename=original_filename,
                original_image_path=str(upload_path),
                original_width=original_width,
                original_height=original_height,
            )
            db.session.add(image_record)
            db.session.commit()

            try:
                vehicle_count, analysis, result_name, model_width, model_height = call_local_yolo(
                    upload_path,
                    image_record.id,
                    {"provider": "local_yolo_vehicle"},
                    draw_vehicle_boxes,
                    class_ids=VEHICLE_CLASS_IDS,
                    target_label="vehicle",
                    count_label="\u8f66\u8f86",
                )
                image_record.model_image_path = str(upload_path)
                image_record.model_width = model_width
                image_record.model_height = model_height
                db.session.commit()
                result_image_url = url_for("static", filename=f"results/{result_name}")
            except Exception as exc:
                error = f"\u8f66\u8f86\u68c0\u6d4b\u5931\u8d25\uff1a{exc}"
                logger.exception("\u8f66\u8f86\u68c0\u6d4b\u8fc7\u7a0b\u4e2d\u51fa\u9519")
                db.session.rollback()

    return render_template(
        "vehicle.html",
        error=error,
        vehicle_count=vehicle_count,
        analysis=analysis,
        result_image_url=result_image_url,
        original_filename=original_filename,
    )


@app.route("/api/cleanup-old-files", methods=["POST"])
def cleanup_old_files():
    max_age_days = int(request.args.get("max_age_days", "7"))
    cutoff = datetime.utcnow() - timedelta(days=max_age_days)
    cleaned = 0

    for record in ImageRecord.query.filter(ImageRecord.uploaded_at < cutoff).all():
        orig_path = Path(record.original_image_path)
        if orig_path.exists():
            orig_path.unlink()
            cleaned += 1
        db.session.delete(record)

    db.session.commit()
    return jsonify({"cleaned_records": cleaned})


@app.route("/progress/detect")
def progress_detect():
    import time

    def generate():
        steps = [
            {"step": 1, "message": "\u6b63\u5728\u52a0\u8f7d YOLO \u6a21\u578b..."},
            {"step": 2, "message": "\u6b63\u5728\u6267\u884c YOLO \u68c0\u6d4b..."},
            {"step": 3, "message": "\u6b63\u5728\u7ed8\u5236\u68c0\u6d4b\u7ed3\u679c..."},
            {"step": 4, "message": "\u68c0\u6d4b\u5b8c\u6210"},
        ]
        for s in steps:
            yield f"data: {json.dumps(s, ensure_ascii=False)}\n\n"
            time.sleep(0.5)

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


@app.route("/api/history/delete/<int:record_id>", methods=["DELETE"])
def delete_history_record(record_id):
    record = db.session.get(ImageRecord, record_id)
    if record is None:
        return jsonify({"success": False, "error": "Record not found"}), 404

    try:
        orig_path = Path(record.original_image_path)
        if orig_path.exists():
            orig_path.unlink()

        for detection in record.detections:
            if detection.result_image_path:
                res_path = Path(detection.result_image_path)
                if res_path.exists():
                    res_path.unlink()
            if detection.raw_llm_response_log_path:
                log_path = Path(detection.raw_llm_response_log_path)
                if log_path.exists():
                    log_path.unlink()
            db.session.delete(detection)

        db.session.delete(record)
        db.session.commit()
        return jsonify({"success": True, "message": "Record deleted successfully"})
    except Exception as exc:
        db.session.rollback()
        logger.exception(f"Failed to delete record {record_id}")
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/history/batch-delete", methods=["POST"])
def batch_delete_history():
    data = request.get_json(silent=True)
    if not data or "ids" not in data or not isinstance(data["ids"], list):
        return jsonify({"success": False, "error": "Invalid request: ids required"}), 400

    ids = data["ids"]
    if not ids:
        return jsonify({"success": False, "error": "No record IDs provided"}), 400

    deleted = 0
    errors = []
    for record_id in ids:
        record = db.session.get(ImageRecord, record_id)
        if record is None:
            errors.append({"id": record_id, "error": "Not found"})
            continue
        try:
            orig_path = Path(record.original_image_path)
            if orig_path.exists():
                orig_path.unlink()
            for detection in record.detections:
                if detection.result_image_path:
                    res_path = Path(detection.result_image_path)
                    if res_path.exists():
                        res_path.unlink()
                if detection.raw_llm_response_log_path:
                    log_path = Path(detection.raw_llm_response_log_path)
                    if log_path.exists():
                        log_path.unlink()
                db.session.delete(detection)
            db.session.delete(record)
            db.session.commit()
            deleted += 1
        except Exception as exc:
            db.session.rollback()
            logger.exception(f"Failed to delete record {record_id}")
            errors.append({"id": record_id, "error": str(exc)})

    return jsonify({
        "success": True,
        "deleted": deleted,
        "errors": errors,
        "message": f"Successfully deleted {deleted} records"
    })


@app.route("/video")
def video_page():
    return render_template("video.html")


@app.route("/api/video/history")
def video_history():
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 10, type=int)
    per_page = min(per_page, 50)
    pagination = VideoRecord.query.order_by(VideoRecord.uploaded_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    records = []
    for record in pagination.items:
        records.append({
            "id": record.id,
            "original_filename": record.original_filename,
            "uploaded_at": record.uploaded_at.strftime("%Y-%m-%d %H:%M:%S") if record.uploaded_at else "",
            "status": record.status,
            "total_frames": record.total_frames,
            "processed_frames": record.processed_frames,
            "fps": record.fps,
            "duration": record.duration,
            "total_persons": record.total_persons,
            "avg_confidence": record.avg_confidence,
            "video_width": record.video_width,
            "video_height": record.video_height,
            "error_message": record.error_message,
            "has_result": bool(record.processed_video_path and Path(record.processed_video_path).exists()),
        })
    return jsonify({
        "records": records,
        "page": pagination.page,
        "pages": pagination.pages,
        "total": pagination.total,
        "has_prev": pagination.has_prev,
        "has_next": pagination.has_next,
    })


@app.route("/api/video/upload", methods=["POST"])
def video_upload():
    file = request.files.get("video")
    if file is None or file.filename == "":
        return jsonify({"success": False, "error": "Please select a video file"}), 400

    if not allowed_video_file(file.filename):
        return jsonify({
            "success": False,
            "error": "Unsupported format. Supported: " + ", ".join(ALLOWED_VIDEO_EXTENSIONS)
        }), 400

    file.seek(0, 2)
    file_size = file.tell()
    file.seek(0)
    max_bytes = VIDEO_MAX_SIZE * 1024 * 1024
    if file_size > max_bytes:
        return jsonify({
            "success": False,
            "error": f"File too large. Maximum: {VIDEO_MAX_SIZE}MB"
        }), 400

    suffix = Path(file.filename).suffix.lower()
    upload_name = f"{uuid4().hex}{suffix}"
    upload_path = VIDEO_UPLOAD_DIR / upload_name
    file.save(upload_path)

    try:
        video_info = get_video_info(upload_path)
    except Exception as exc:
        upload_path.unlink(missing_ok=True)
        return jsonify({"success": False, "error": f"Cannot read video: {exc}"}), 400

    record = VideoRecord(
        original_filename=file.filename,
        video_path=str(upload_path),
        status="pending",
        total_frames=video_info["total_frames"],
        fps=video_info["fps"],
        duration=video_info["duration"],
        video_width=video_info["width"],
        video_height=video_info["height"],
    )
    db.session.add(record)
    db.session.commit()

    def run_video_processing(video_id):
        with app.app_context():
            try:
                process_video_detection(video_id)
            except Exception as exc:
                logger.exception(f"Fatal error in video processing thread for record {video_id}")
                update_video_progress(video_id, status="failed", message=f"Fatal error: {exc}")
                try:
                    rec = db.session.get(VideoRecord, video_id)
                    if rec:
                        rec.status = "failed"
                        rec.error_message = str(exc)
                        db.session.commit()
                except Exception:
                    db.session.rollback()

    thread = threading.Thread(target=run_video_processing, args=(record.id,), daemon=True)
    thread.start()

    return jsonify({
        "success": True,
        "record_id": record.id,
        "video_info": video_info,
        "message": "Video uploaded and processing started",
    })


@app.route("/api/video/progress/<int:video_id>")
def video_progress(video_id):
    progress = get_video_progress(video_id)
    record = db.session.get(VideoRecord, video_id)
    if record:
        if progress["status"] == "unknown" or progress["status"] == "processing":
            progress["status"] = record.status
        if progress["total_frames"] == 0 and record.total_frames:
            progress["total_frames"] = record.total_frames
        if record.status == "pending":
            progress["progress"] = 0
            progress["current_frame"] = 0
            progress["total_frames"] = record.total_frames or 0
            progress["message"] = "等待处理队列..."
        elif record.status == "failed" and not progress.get("message"):
            progress["message"] = record.error_message or "处理失败"
    return jsonify(progress)


@app.route("/api/video/download/<int:video_id>")
def video_download(video_id):
    record = db.session.get(VideoRecord, video_id)
    if record is None:
        return jsonify({"success": False, "error": "Record not found"}), 404

    if record.status != "completed" or not record.processed_video_path:
        return jsonify({"success": False, "error": "Video not yet processed"}), 400

    video_path = Path(record.processed_video_path)
    if not video_path.exists():
        return jsonify({"success": False, "error": "Processed video file not found"}), 404

    download_name = f"detected_{record.original_filename}"
    return send_file(
        str(video_path),
        as_attachment=True,
        download_name=download_name,
        mimetype="video/mp4",
    )


@app.route("/api/video/result/<int:video_id>")
def video_result_url(video_id):
    record = db.session.get(VideoRecord, video_id)
    if record is None:
        return jsonify({"success": False, "error": "Record not found"}), 404

    if record.status != "completed" or not record.processed_video_path:
        return jsonify({"success": False, "error": "Video not yet processed"}), 400

    video_path = Path(record.processed_video_path)
    if not video_path.exists():
        return jsonify({"success": False, "error": "Processed video file not found"}), 404

    static_dir = BASE_DIR / "static"
    relative_path = video_path.relative_to(static_dir)
    url = url_for("static", filename=str(relative_path.as_posix()))
    progress = get_video_progress(video_id)
    metadata = load_video_count_metadata(video_path)
    person_counts = metadata["person_counts"]
    first_frame_count = person_counts[0] if person_counts else (progress.get("current_person_count") or record.total_persons or 0)
    return jsonify({
        "success": True,
        "url": url,
        "stream_url": url_for("video_stream", video_id=video_id),
        "frame_url": url_for("video_frame", video_id=video_id),
        "filename": record.original_filename,
        "current_person_count": first_frame_count,
        "total_persons": record.total_persons or 0,
    })


@app.route("/api/video/frame/<int:video_id>")
def video_frame(video_id):
    record = db.session.get(VideoRecord, video_id)
    if record is None:
        return jsonify({"success": False, "error": "Record not found"}), 404

    if record.status != "completed" or not record.processed_video_path:
        return jsonify({"success": False, "error": "Video not yet processed"}), 400

    video_path = Path(record.processed_video_path)
    if not video_path.exists():
        return jsonify({"success": False, "error": "Processed video file not found"}), 404

    import cv2

    metadata = load_video_count_metadata(video_path)
    person_counts = metadata["person_counts"]
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return jsonify({"success": False, "error": "Cannot open processed video"}), 500

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or record.processed_frames or 0)
        if total_frames <= 0:
            return jsonify({"success": False, "error": "Processed video has no frames"}), 404

        frame_index = request.args.get("frame", 0, type=int)
        frame_index = max(0, min(frame_index, total_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            return jsonify({"success": False, "error": "Cannot read requested frame"}), 500

        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            return jsonify({"success": False, "error": "Cannot encode requested frame"}), 500

        fps = cap.get(cv2.CAP_PROP_FPS) or record.fps or 25
        person_count = person_counts[frame_index] if frame_index < len(person_counts) else record.total_persons or 0
        response = Response(buffer.tobytes(), mimetype="image/jpeg")
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Person-Count"] = str(person_count)
        response.headers["X-Frame-Index"] = str(frame_index)
        response.headers["X-Frame-Total"] = str(total_frames)
        response.headers["X-Fps"] = str(fps)
        return response
    finally:
        cap.release()


@app.route("/api/video/stream/<int:video_id>")
def video_stream(video_id):
    record = db.session.get(VideoRecord, video_id)
    if record is None:
        return jsonify({"success": False, "error": "Record not found"}), 404

    if record.status != "completed" or not record.processed_video_path:
        return jsonify({"success": False, "error": "Video not yet processed"}), 400

    video_path = Path(record.processed_video_path)
    if not video_path.exists():
        return jsonify({"success": False, "error": "Processed video file not found"}), 404

    def generate_frames():
        import time
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        delay = min(max(1 / fps, 0.01), 0.08)
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                if not ok:
                    continue
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + buffer.tobytes()
                    + b"\r\n"
                )
                time.sleep(delay)
        finally:
            cap.release()

    return Response(
        stream_with_context(generate_frames()),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/video/delete/<int:video_id>", methods=["DELETE"])
def delete_video_record(video_id):
    record = db.session.get(VideoRecord, video_id)
    if record is None:
        return jsonify({"success": False, "error": "Record not found"}), 404

    try:
        video_path = Path(record.video_path)
        if video_path.exists():
            video_path.unlink()
        if record.processed_video_path:
            result_path = Path(record.processed_video_path)
            if result_path.exists():
                result_path.unlink()
            metadata_path = get_video_count_metadata_path(result_path)
            if metadata_path.exists():
                metadata_path.unlink()
        db.session.delete(record)
        db.session.commit()
        return jsonify({"success": True, "message": "Video record deleted"})
    except Exception as exc:
        db.session.rollback()
        logger.exception(f"Failed to delete video record {video_id}")
        return jsonify({"success": False, "error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
