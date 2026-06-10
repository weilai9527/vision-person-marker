import json
import asyncio
import logging
import mimetypes
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from uuid import uuid4
from filelock import FileLock

from flask import Flask, render_template, request, url_for, Response, jsonify, stream_with_context, send_file
from PIL import Image
from sqlalchemy import text

from services.config_service import load_api_config, load_api_config_for_display, save_api_config, mask_api_key, is_masked_api_key, normalize_chat_endpoint, infer_provider, normalize_model_name
from services.dashboard_service import build_dashboard_stats
from services.camera_engine import process_camera_frame_image
from services.webrtc_engine import get_camera_webrtc_loop, create_camera_webrtc_answer, close_camera_webrtc_session, get_webrtc_result

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
from models import db, DetectionResult, ImageRecord, VideoRecord
from services.image_service import draw_person_boxes, draw_vehicle_boxes
from services.llm_service import call_vehicle_yolo_judge_model, call_yolo_judge_model, call_vision_model, call_vehicle_vision_model
from services.yolo_service import DEFAULT_PERSON_CLASS_IDS, VEHICLE_CLASS_IDS, call_local_yolo, get_available_yolo_models, get_default_yolo_model_name, is_complex_detection_scene, is_suspected_yolo_miss, normalize_yolo_model_name, run_yolo_detection
from services.video_service import (
    process_video_detection,
    get_video_progress,
    update_video_progress,
    allowed_video_file,
    get_video_info,
    get_video_count_metadata_path,
    load_video_count_metadata,
)
from services.export_service import (
    EXPORT_FORMATS, IMAGE_FIELD_MAP, VIDEO_FIELD_MAP,
    create_export_task, run_export_task, get_task,
    get_export_history,
)
from services.history_service import (
    query_image_records, query_video_records,
    serialize_image_record, serialize_video_record,
    delete_image_record, batch_delete_image_records,
    delete_video_record, batch_delete_video_records,
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


def create_export_workbook():
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font

    return Workbook(), Font, Alignment


def ensure_video_record_columns() -> None:
    columns = {
        row[1]
        for row in db.session.execute(text("PRAGMA table_info(video_record)")).fetchall()
    }
    if "detection_target" not in columns:
        db.session.execute(
            text("ALTER TABLE video_record ADD COLUMN detection_target VARCHAR(20) NOT NULL DEFAULT 'person'")
        )
        db.session.commit()
    if "unique_count" not in columns:
        db.session.execute(
            text("ALTER TABLE video_record ADD COLUMN unique_count INTEGER NOT NULL DEFAULT 0")
        )
        db.session.commit()
    if "sum_count" not in columns:
        db.session.execute(
            text("ALTER TABLE video_record ADD COLUMN sum_count INTEGER NOT NULL DEFAULT 0")
        )
        db.session.commit()
    if "yolo_model_name" not in columns:
        db.session.execute(
            text("ALTER TABLE video_record ADD COLUMN yolo_model_name VARCHAR(100) NOT NULL DEFAULT ''")
        )
        db.session.commit()

    detection_columns = {
        row[1]
        for row in db.session.execute(text("PRAGMA table_info(detection_result)")).fetchall()
    }
    detection_migrations = {
        "raw_yolo_boxes_json": "TEXT",
        "llm_boxes_json": "TEXT",
        "final_source": "VARCHAR(50)",
        "review_status": "VARCHAR(20) DEFAULT 'pending'",
        "detection_strategy": "VARCHAR(50)",
        "yolo_miss_reason": "VARCHAR(100)",
    }
    for col_name, col_type in detection_migrations.items():
        if col_name not in detection_columns:
            db.session.execute(
                text(f"ALTER TABLE detection_result ADD COLUMN {col_name} {col_type}")
            )
            db.session.commit()


with app.app_context():
    db.session.execute(text("PRAGMA journal_mode=WAL"))
    db.session.execute(text("PRAGMA synchronous=NORMAL"))
    db.session.commit()
    db.create_all()
    ensure_video_record_columns()

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_RESULT_DIR.mkdir(parents=True, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


@app.context_processor
def inject_globals():
    return {
        "api_providers": API_PROVIDERS,
        "api_config": load_api_config_for_display(),
        "has_api_key": bool(load_api_config().get("api_key")),
        "yolo_models": get_available_yolo_models(),
        "default_yolo_model_name": get_default_yolo_model_name(),
    }


_IMAGE_TARGET_CONFIGS = {
    "person": {
        "class_ids": DEFAULT_PERSON_CLASS_IDS,
        "target_label": "person",
        "complex_label": "行人",
        "count_label": "行人",
        "count_key": "person_count",
        "draw_boxes": draw_person_boxes,
        "yolo_kwargs": {},
        "judge_model": call_yolo_judge_model,
        "vision_model": call_vision_model,
        "fallback_provider": "local_yolo",
        "error_prefix": "检测失败",
    },
    "vehicle": {
        "class_ids": VEHICLE_CLASS_IDS,
        "target_label": "vehicle",
        "complex_label": "车辆",
        "count_label": "车辆",
        "count_key": "vehicle_count",
        "draw_boxes": draw_vehicle_boxes,
        "yolo_kwargs": {"conf": 0.18, "iou": 0.55, "imgsz": 1536, "min_area": 20},
        "judge_model": call_vehicle_yolo_judge_model,
        "vision_model": call_vehicle_vision_model,
        "fallback_provider": "local_yolo_vehicle",
        "error_prefix": "车辆检测失败",
    },
}


def _handle_image_upload(req, template_name: str, target_type: str = "person"):
    cfg = _IMAGE_TARGET_CONFIGS[target_type]
    error = None
    success = None
    detection_count = None
    analysis = None
    result_image_url = None
    original_filename = None
    api_config = load_api_config_for_display()
    selected_yolo_model = get_default_yolo_model_name()

    if req.method == "POST":
        action = req.form.get("action")
        if action == "save_api":
            provider = req.form.get("provider", "custom")
            api_url = req.form.get("api_url", "")
            api_key = req.form.get("api_key", "")
            model = req.form.get("model", "")
            current_api_config = load_api_config()
            submitted_api_key = api_key.strip()
            if (not submitted_api_key or is_masked_api_key(submitted_api_key)) and not current_api_config.get("api_key"):
                error = "请填写完整 API Key，不能使用已加密的 Key。"
            else:
                save_api_config(provider, api_url, api_key, model)
                api_config = load_api_config_for_display()
                success = "API 配置已保存。"
        else:
            selected_yolo_model = normalize_yolo_model_name(req.form.get("yolo_model") or req.form.get("model_name"))
            file = req.files.get("image")
            if file is None or file.filename == "":
                error = "请先选择一张图片。"
            elif not allowed_file(file.filename):
                error = "只支持 jpg、jpeg、png、bmp、webp 格式。"
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
                        preflight_boxes, preflight_width, preflight_height = run_yolo_detection(
                            upload_path, class_ids=cfg["class_ids"], target_label=cfg["target_label"],
                            model_name=selected_yolo_model, **cfg["yolo_kwargs"],
                        )

                        should_llm_detect, detect_reason = is_suspected_yolo_miss(
                            preflight_boxes, preflight_width, preflight_height, target_label=cfg["complex_label"]
                        )

                        is_complex, complexity_reason = is_complex_detection_scene(
                            preflight_boxes, preflight_width, preflight_height, target_label=cfg["complex_label"]
                        )

                        if should_llm_detect:
                            try:
                                detection_count, analysis, result_name, model_width, model_height = cfg["vision_model"](
                                    upload_path,
                                    image_record.id,
                                    detection_config,
                                    cfg["draw_boxes"],
                                    analysis_prefix=f"YOLO 疑似漏检：{detect_reason}，已交给大模型补检。",
                                )
                            except Exception as llm_exc:
                                detection_count, analysis, result_name, model_width, model_height = call_local_yolo(
                                    upload_path,
                                    image_record.id,
                                    {"provider": "auto_yolo_llm_detect_fallback"},
                                    cfg["draw_boxes"],
                                    class_ids=cfg["class_ids"],
                                    target_label=cfg["target_label"],
                                    count_label=cfg["count_label"],
                                    analysis_prefix=(
                                        f"YOLO 疑似漏检：{detect_reason}，但大模型补检失败，已回退本地 YOLO。"
                                        f"大模型错误：{llm_exc}。"
                                    ),
                                    precomputed_boxes=preflight_boxes,
                                    precomputed_size=(preflight_width, preflight_height),
                                    model_name=selected_yolo_model,
                                    raw_yolo_boxes=preflight_boxes,
                                    detection_strategy="llm_detect_fallback",
                                    yolo_miss_reason=detect_reason,
                                )
                        elif is_complex:
                            try:
                                detection_count, analysis, result_name, model_width, model_height = cfg["judge_model"](
                                    upload_path,
                                    image_record.id,
                                    detection_config,
                                    cfg["draw_boxes"],
                                    preflight_boxes,
                                    preflight_width,
                                    preflight_height,
                                    analysis_prefix=f"YOLO 快速预判：{complexity_reason}，已交给大模型裁判复核。",
                                )
                            except Exception as llm_exc:
                                detection_count, analysis, result_name, model_width, model_height = call_local_yolo(
                                    upload_path,
                                    image_record.id,
                                    {"provider": "auto_yolo_llm_fallback"},
                                    cfg["draw_boxes"],
                                    class_ids=cfg["class_ids"],
                                    target_label=cfg["target_label"],
                                    count_label=cfg["count_label"],
                                    analysis_prefix=(
                                        f"YOLO 快速预判：{complexity_reason}，但大模型调用失败，已回退本地 YOLO。"
                                        f"大模型错误：{llm_exc}。"
                                    ),
                                    precomputed_boxes=preflight_boxes,
                                    precomputed_size=(preflight_width, preflight_height),
                                    model_name=selected_yolo_model,
                                    raw_yolo_boxes=preflight_boxes,
                                    detection_strategy="yolo_judge_fallback",
                                )
                        else:
                            detection_count, analysis, result_name, model_width, model_height = call_local_yolo(
                                upload_path,
                                image_record.id,
                                {"provider": "auto_yolo_simple"},
                                cfg["draw_boxes"],
                                class_ids=cfg["class_ids"],
                                target_label=cfg["target_label"],
                                count_label=cfg["count_label"],
                                analysis_prefix=f"YOLO 快速预判：{complexity_reason}，直接使用本地结果。",
                                precomputed_boxes=preflight_boxes,
                                precomputed_size=(preflight_width, preflight_height),
                                model_name=selected_yolo_model,
                                raw_yolo_boxes=preflight_boxes,
                                detection_strategy="yolo_only",
                            )
                    else:
                        detection_count, analysis, result_name, model_width, model_height = call_local_yolo(
                            upload_path, image_record.id, {"provider": cfg["fallback_provider"]}, cfg["draw_boxes"],
                            class_ids=cfg["class_ids"], target_label=cfg["target_label"],
                            count_label=cfg["count_label"], model_name=selected_yolo_model
                        )
                    image_record.model_image_path = f"static/results/{result_name}"
                    image_record.model_width = model_width
                    image_record.model_height = model_height
                    db.session.commit()
                    result_image_url = url_for("static", filename=f"results/{result_name}")
                except Exception as exc:
                    error = f"{cfg['error_prefix']}：{exc}"
                    logger.exception(f"{cfg['error_prefix']}过程中出错")
                    db.session.rollback()

    extra_context = {}
    if template_name == "index.html":
        extra_context["dashboard_stats"] = build_dashboard_stats()

    return render_template(
        template_name,
        error=error,
        success=success,
        api_config=api_config,
        api_providers=API_PROVIDERS,
        has_api_key=bool(load_api_config().get("api_key")),
        person_count=detection_count if target_type == "person" else None,
        vehicle_count=detection_count if target_type == "vehicle" else None,
        analysis=analysis,
        result_image_url=result_image_url,
        original_filename=original_filename,
        selected_yolo_model=selected_yolo_model,
        **extra_context,
    )


@app.route("/", methods=["GET", "POST"])
def index():
    return _handle_image_upload(request, "index.html", "person")


@app.route("/person-image", methods=["GET", "POST"])
def person_image_page():
    return _handle_image_upload(request, "person_image.html", "person")


@app.route("/history")
def history():
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 10, type=int)
    per_page = min(per_page, 50)
    pagination = ImageRecord.query.order_by(ImageRecord.uploaded_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    return render_template("history.html", pagination=pagination, records=pagination.items)


@app.route("/history/image/pedestrian")
def history_image_pedestrian():
    return render_template("history_image_pedestrian.html",
                           history_type="image_pedestrian",
                           type_label="图片行人检测历史",
                           detection_label="行人",
                           media_label="图片")


@app.route("/history/image/vehicle")
def history_image_vehicle():
    return render_template("history_image_vehicle.html",
                           history_type="image_vehicle",
                           type_label="图片车辆检测历史",
                           detection_label="车辆",
                           media_label="图片")


@app.route("/history/video/pedestrian")
def history_video_pedestrian():
    return render_template("history_video_pedestrian.html",
                           history_type="video_pedestrian",
                           type_label="视频行人检测历史",
                           detection_label="行人",
                           media_label="视频")


@app.route("/history/video/vehicle")
def history_video_vehicle():
    return render_template("history_video_vehicle.html",
                           history_type="video_vehicle",
                           type_label="视频车辆检测历史",
                           detection_label="车辆",
                           media_label="视频")


@app.route("/data-export")
def data_export():
    return render_template("data_export.html")


@app.route("/export/image/pedestrian")
def export_image_pedestrian():
    return render_template("export_image_pedestrian.html",
                           export_formats=EXPORT_FORMATS,
                           field_map=IMAGE_FIELD_MAP,
                           export_type="image_pedestrian",
                           export_type_label="图片行人检测数据导出",
                           detection_label="行人",
                           media_label="图片")


@app.route("/export/image/vehicle")
def export_image_vehicle():
    return render_template("export_image_vehicle.html",
                           export_formats=EXPORT_FORMATS,
                           field_map=IMAGE_FIELD_MAP,
                           export_type="image_vehicle",
                           export_type_label="图片车辆检测数据导出",
                           detection_label="车辆",
                           media_label="图片")


@app.route("/export/video/pedestrian")
def export_video_pedestrian():
    return render_template("export_video_pedestrian.html",
                           export_formats=EXPORT_FORMATS,
                           field_map=VIDEO_FIELD_MAP,
                           export_type="video_pedestrian",
                           export_type_label="视频行人检测数据导出",
                           detection_label="行人",
                           media_label="视频")


@app.route("/export/video/vehicle")
def export_video_vehicle():
    return render_template("export_video_vehicle.html",
                           export_formats=EXPORT_FORMATS,
                           field_map=VIDEO_FIELD_MAP,
                           export_type="video_vehicle",
                           export_type_label="视频车辆检测数据导出",
                           detection_label="车辆",
                           media_label="视频")


@app.route("/camera")
def camera_page():
    return render_template("camera.html")


@app.route("/api/yolo/models")
def yolo_models_api():
    return jsonify(
        {
            "success": True,
            "models": get_available_yolo_models(),
            "current": get_default_yolo_model_name(),
        }
    )


@app.route("/api/camera/detect", methods=["POST"])
def camera_detect():
    frame = request.files.get("frame")
    if frame is None or frame.filename == "":
        return jsonify({"success": False, "error": "No camera frame provided"}), 400

    target = request.form.get("target", "both")
    target = target if target in {"person", "vehicle", "both"} else "both"
    session_id = (request.form.get("session_id") or request.remote_addr or "default").strip()[:80]
    reset_tracking = request.form.get("reset_tracking") == "1"
    model_name = request.form.get("model_name") or request.form.get("yolo_model")

    try:
        with Image.open(frame.stream) as image:
            result = process_camera_frame_image(image, target, session_id, reset_tracking, model_name)
            result["transport"] = "http"
    except Exception as exc:
        logger.exception("摄像头实时检测失败")
        return jsonify({"success": False, "error": str(exc)}), 500

    return jsonify(result)


@app.route("/api/camera/webrtc/offer", methods=["POST"])
def camera_webrtc_offer():
    from services.webrtc_engine import RTCPeerConnection, RTCSessionDescription
    if RTCPeerConnection is None or RTCSessionDescription is None:
        return jsonify({
            "success": False,
            "error": "WebRTC backend is not installed. Please install aiortc.",
        }), 501

    data = request.get_json(silent=True) or {}
    if not data.get("sdp") or not data.get("type"):
        return jsonify({"success": False, "error": "Invalid WebRTC offer"}), 400

    target = data.get("target", "both")
    target = target if target in {"person", "vehicle", "both"} else "both"
    session_id = (data.get("session_id") or request.remote_addr or "default").strip()[:80]
    reset_tracking = bool(data.get("reset_tracking"))
    model_name = normalize_yolo_model_name(data.get("model_name") or data.get("yolo_model"))

    try:
        loop = get_camera_webrtc_loop()
        future = asyncio.run_coroutine_threadsafe(
            create_camera_webrtc_answer(data, session_id, target, reset_tracking, model_name),
            loop,
        )
        answer = future.result(timeout=15)
    except Exception as exc:
        logger.exception("WebRTC offer failed")
        return jsonify({"success": False, "error": str(exc)}), 500

    return jsonify({"success": True, "answer": answer, "session_id": session_id})


@app.route("/api/camera/webrtc/result")
def camera_webrtc_result():
    session_id = (request.args.get("session_id") or request.remote_addr or "default").strip()[:80]
    result = get_webrtc_result(session_id)
    if result is None:
        return jsonify({"success": True, "pending": True, "transport": "webrtc"})
    return jsonify(result)


@app.route("/api/camera/webrtc/stop", methods=["POST"])
def camera_webrtc_stop():
    data = request.get_json(silent=True) or {}
    session_id = (data.get("session_id") or request.remote_addr or "default").strip()[:80]
    try:
        loop = get_camera_webrtc_loop()
        future = asyncio.run_coroutine_threadsafe(close_camera_webrtc_session(session_id), loop)
        future.result(timeout=5)
    except Exception as exc:
        logger.exception("WebRTC stop failed")
        return jsonify({"success": False, "error": str(exc)}), 500
    return jsonify({"success": True})


_IMAGE_EXPORT_COL_WIDTHS = {
    "A": 8, "B": 32, "C": 20, "D": 14, "E": 14,
    "F": 8, "G": 20, "H": 14, "I": 60, "J": 18, "K": 18
}

_VIDEO_EXPORT_COL_WIDTHS = {
    "A": 8, "B": 32, "C": 20, "D": 12, "E": 10,
    "F": 10, "G": 10, "H": 14, "I": 8, "J": 12,
    "K": 18, "L": 18, "M": 14, "N": 40
}

def _generate_excel_response(ws_title: str, headers: list, rows: list, column_widths: dict, filename_prefix: str):
    wb, Font, Alignment = create_export_workbook()
    ws = wb.active
    ws.title = ws_title

    ws.append(headers)
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center")

    for row in rows:
        ws.append(row)

    for col_letter, width in column_widths.items():
        ws.column_dimensions[col_letter].width = width

    buffer = BytesIO()
    wb.save(buffer)
    buffer.seek(0)
    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.route("/api/history/export")
def export_history():
    headers = [
        "\u8bb0\u5f55ID", "\u539f\u59cb\u6587\u4ef6\u540d", "\u4e0a\u4f20\u65f6\u95f4",
        "\u539f\u59cb\u5c3a\u5bf8", "\u6a21\u578b\u5c3a\u5bf8", "\u68c0\u6d4bID",
        "\u68c0\u6d4b\u65f6\u95f4", "\u68c0\u6d4b\u4eba\u6570", "\u6a21\u578b\u8bf4\u660e",
        "API\u63d0\u4f9b\u5546", "\u6a21\u578b\u540d\u79f0",
    ]
    records = ImageRecord.query.order_by(ImageRecord.uploaded_at.desc()).all()
    rows = []
    for record in records:
        base = [
            record.id,
            record.original_filename,
            record.uploaded_at.strftime("%Y-%m-%d %H:%M:%S") if record.uploaded_at else "",
            f"{record.original_width or '-'}x{record.original_height or '-'}",
            f"{record.model_width or '-'}x{record.model_height or '-'}",
        ]
        if record.detections:
            for detection in record.detections:
                rows.append(base + [
                    detection.id,
                    detection.detected_at.strftime("%Y-%m-%d %H:%M:%S") if detection.detected_at else "",
                    detection.person_count,
                    detection.llm_analysis_text or "",
                    detection.llm_api_provider or "",
                    detection.llm_model_name or "",
                ])
        else:
            rows.append(base + ["", "", "", "", "", ""])

    return _generate_excel_response(
        ws_title="\u56fe\u7247\u68c0\u6d4b\u5386\u53f2",
        headers=headers,
        rows=rows,
        column_widths=_IMAGE_EXPORT_COL_WIDTHS,
        filename_prefix="\u56fe\u7247\u68c0\u6d4b\u5386\u53f2"
    )


@app.route("/api/video/export")
def export_video_history():
    headers = [
        "\u8bb0\u5f55ID", "\u539f\u59cb\u6587\u4ef6\u540d", "\u4e0a\u4f20\u65f6\u95f4",
        "\u68c0\u6d4b\u76ee\u6807", "\u72b6\u6001", "\u603b\u5e27\u6570", "\u5904\u7406\u5e27\u6570",
        "\u89c6\u9891\u5c3a\u5bf8", "FPS", "\u65f6\u957f(\u79d2)", "\u6700\u5927\u4eba\u6570/\u8f66\u8f86\u6570",
        "\u53bb\u91cd\u540e\u603b\u6570", "\u5e73\u5747\u7f6e\u4fe1\u5ea6", "\u9519\u8bef\u4fe1\u606f",
    ]
    records = VideoRecord.query.order_by(VideoRecord.uploaded_at.desc()).all()
    rows = [[
        record.id,
        record.original_filename,
        record.uploaded_at.strftime("%Y-%m-%d %H:%M:%S") if record.uploaded_at else "",
        record.detection_target or "person",
        record.status or "",
        record.total_frames,
        record.processed_frames,
        f"{record.video_width or '-'}x{record.video_height or '-'}",
        round(record.fps, 2) if record.fps else "",
        round(record.duration, 2) if record.duration else "",
        record.total_persons,
        record.unique_count or record.total_persons or 0,
        round(record.avg_confidence, 4) if record.avg_confidence else "",
        record.error_message or "",
    ] for record in records]

    return _generate_excel_response(
        ws_title="\u89c6\u9891\u68c0\u6d4b\u5386\u53f2",
        headers=headers,
        rows=rows,
        column_widths=_VIDEO_EXPORT_COL_WIDTHS,
        filename_prefix="\u89c6\u9891\u68c0\u6d4b\u5386\u53f2"
    )


@app.route("/api/vehicle/video/export")
def export_vehicle_video_history():
    headers = [
        "\u8bb0\u5f55ID", "\u539f\u59cb\u6587\u4ef6\u540d", "\u4e0a\u4f20\u65f6\u95f4",
        "\u68c0\u6d4b\u76ee\u6807", "\u72b6\u6001", "\u603b\u5e27\u6570", "\u5904\u7406\u5e27\u6570",
        "\u89c6\u9891\u5c3a\u5bf8", "FPS", "\u65f6\u957f(\u79d2)", "\u6700\u5927\u8f66\u8f86\u6570",
        "\u53bb\u91cd\u540e\u603b\u6570", "\u5e73\u5747\u7f6e\u4fe1\u5ea6", "\u9519\u8bef\u4fe1\u606f",
    ]
    records = VideoRecord.query.filter_by(detection_target="vehicle").order_by(VideoRecord.uploaded_at.desc()).all()
    rows = [[
        record.id,
        record.original_filename,
        record.uploaded_at.strftime("%Y-%m-%d %H:%M:%S") if record.uploaded_at else "",
        record.detection_target or "vehicle",
        record.status or "",
        record.total_frames,
        record.processed_frames,
        f"{record.video_width or '-'}x{record.video_height or '-'}",
        round(record.fps, 2) if record.fps else "",
        round(record.duration, 2) if record.duration else "",
        record.total_persons,
        record.unique_count or record.total_persons or 0,
        round(record.avg_confidence, 4) if record.avg_confidence else "",
        record.error_message or "",
    ] for record in records]

    return _generate_excel_response(
        ws_title="\u8f66\u8f86\u89c6\u9891\u68c0\u6d4b\u5386\u53f2",
        headers=headers,
        rows=rows,
        column_widths=_VIDEO_EXPORT_COL_WIDTHS,
        filename_prefix="\u8f66\u8f86\u89c6\u9891\u68c0\u6d4b\u5386\u53f2"
    )


@app.route("/api/vehicle/image/export")
def export_vehicle_image_history():
    headers = [
        "\u8bb0\u5f55ID", "\u539f\u59cb\u6587\u4ef6\u540d", "\u4e0a\u4f20\u65f6\u95f4",
        "\u539f\u59cb\u5c3a\u5bf8", "\u6a21\u578b\u5c3a\u5bf8", "\u68c0\u6d4bID",
        "\u68c0\u6d4b\u65f6\u95f4", "\u68c0\u6d4b\u8f66\u8f86\u6570", "\u6a21\u578b\u8bf4\u660e",
        "API\u63d0\u4f9b\u5546", "\u6a21\u578b\u540d\u79f0",
    ]
    records = ImageRecord.query.join(DetectionResult).filter(
        DetectionResult.llm_api_provider == "local_yolo_vehicle"
    ).order_by(ImageRecord.uploaded_at.desc()).all()
    rows = []
    for record in records:
        base = [
            record.id,
            record.original_filename,
            record.uploaded_at.strftime("%Y-%m-%d %H:%M:%S") if record.uploaded_at else "",
            f"{record.original_width or '-'}x{record.original_height or '-'}",
            f"{record.model_width or '-'}x{record.model_height or '-'}",
        ]
        vehicle_detections = [d for d in record.detections if d.llm_api_provider == "local_yolo_vehicle"]
        if vehicle_detections:
            for detection in vehicle_detections:
                rows.append(base + [
                    detection.id,
                    detection.detected_at.strftime("%Y-%m-%d %H:%M:%S") if detection.detected_at else "",
                    detection.person_count,
                    detection.llm_analysis_text or "",
                    detection.llm_api_provider or "",
                    detection.llm_model_name or "",
                ])
        else:
            rows.append(base + ["", "", "", "", "", ""])

    return _generate_excel_response(
        ws_title="\u8f66\u8f86\u56fe\u7247\u68c0\u6d4b\u5386\u53f2",
        headers=headers,
        rows=rows,
        column_widths=_IMAGE_EXPORT_COL_WIDTHS,
        filename_prefix="\u8f66\u8f86\u56fe\u7247\u68c0\u6d4b\u5386\u53f2"
    )


def _handle_vehicle_image_upload(req, template_name: str):
    return _handle_image_upload(req, template_name, "vehicle")


@app.route("/vehicle", methods=["GET", "POST"])
def vehicle_page():
    return _handle_image_upload(request, "vehicle.html", "vehicle")


@app.route("/vehicle-image", methods=["GET", "POST"])
def vehicle_image_page():
    return _handle_image_upload(request, "vehicle_image.html", "vehicle")


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

        # 同步清理检测结果图和 LLM 日志，防止静态目录膨胀
        for detection in record.detections:
            if detection.result_image_path:
                res_path = Path(detection.result_image_path)
                if res_path.exists():
                    res_path.unlink()
            if detection.raw_llm_response_log_path:
                log_path = Path(detection.raw_llm_response_log_path)
                if log_path.exists():
                    log_path.unlink()

        db.session.delete(record)

    # 同步清理过期视频记录及关联文件
    for record in VideoRecord.query.filter(VideoRecord.uploaded_at < cutoff).all():
        video_path = Path(record.video_path)
        if video_path.exists():
            video_path.unlink()
            cleaned += 1
        if record.processed_video_path:
            processed_path = Path(record.processed_video_path)
            if processed_path.exists():
                processed_path.unlink()
            metadata_path = processed_path.with_suffix(".counts.json")
            if metadata_path.exists():
                metadata_path.unlink()
        db.session.delete(record)

    db.session.commit()
    return jsonify({"cleaned_records": cleaned})


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


def video_history_response(detection_target: str):
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 10, type=int)
    per_page = min(per_page, 50)
    pagination = VideoRecord.query.filter_by(detection_target=detection_target).order_by(VideoRecord.uploaded_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    return jsonify({
        "records": [serialize_video_record(record) for record in pagination.items],
        "page": pagination.page,
        "pages": pagination.pages,
        "total": pagination.total,
        "has_prev": pagination.has_prev,
        "has_next": pagination.has_next,
    })


@app.route("/api/video/history")
def video_history():
    return video_history_response("person")


@app.route("/api/vehicle/video/history")
def vehicle_video_history():
    return video_history_response("vehicle")


# 全局视频处理线程池，max_workers=1 保证视频排队串行处理，彻底杜绝并发导致 GPU 显存或 CPU 内存溢出(OOM)
video_task_executor = ThreadPoolExecutor(max_workers=1)


def start_video_processing_thread(record_id: int) -> None:
    def run_video_processing(video_id):
        with app.app_context():
            # 在 instance 目录下创建一个专用的视频排队锁文件
            lock_path = DATABASE_DIR / "video_processing.lock"
            try:
                # 获取跨进程文件锁，确保多个 Worker 下也能严格串行执行 YOLO 视频推理
                with FileLock(str(lock_path)):
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

    video_task_executor.submit(run_video_processing, record_id)


def handle_video_upload(detection_target: str):
    selected_yolo_model = normalize_yolo_model_name(request.form.get("yolo_model") or request.form.get("model_name"))
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
        detection_target=detection_target,
        yolo_model_name=selected_yolo_model,
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
    start_video_processing_thread(record.id)

    return jsonify({
        "success": True,
        "record_id": record.id,
        "video_info": video_info,
        "model_name": selected_yolo_model,
        "message": "Video uploaded and processing started",
    })


@app.route("/api/video/upload", methods=["POST"])
def video_upload():
    return handle_video_upload("person")


@app.route("/api/vehicle/video/upload", methods=["POST"])
def vehicle_video_upload():
    return handle_video_upload("vehicle")


@app.route("/api/video/progress/<int:video_id>")
def video_progress(video_id):
    progress = get_video_progress(video_id)
    record = db.session.get(VideoRecord, video_id)
    if record:
        progress["detection_target"] = record.detection_target or "person"
        if progress["status"] == "unknown" or progress["status"] == "processing":
            progress["status"] = record.status
        if progress["total_frames"] == 0 and record.total_frames:
            progress["total_frames"] = record.total_frames
        if record.status == "pending":
            progress["progress"] = 0
            progress["current_frame"] = 0
            progress["total_frames"] = record.total_frames or 0
            progress["message"] = "等待处理队列..."
        elif record.status == "completed":
            progress["total_persons"] = record.total_persons or 0
            progress["total_count"] = record.unique_count or record.total_persons or 0
            progress["sum_count"] = record.sum_count or 0
            progress["unique_count"] = record.unique_count or 0
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
    mime_type = mimetypes.guess_type(str(video_path))[0] or "video/mp4"
    return send_file(
        str(video_path),
        as_attachment=True,
        download_name=download_name,
        mimetype=mime_type,
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
        "detection_target": record.detection_target,
        "current_person_count": first_frame_count,
        "current_count": first_frame_count,
        "total_persons": record.total_persons or 0,
        "total_count": record.unique_count or record.total_persons or 0,
        "unique_count": metadata.get("unique_count") or record.unique_count or 0,
        "sum_count": metadata.get("sum_count") or record.sum_count or 0,
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
        response.headers["X-Detection-Count"] = str(person_count)
        response.headers["X-Detection-Target"] = record.detection_target or "person"
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
        if fps <= 0:
            fps = 25
        frame_duration = 1.0 / fps

        try:
            start_time = time.perf_counter()
            frame_count = 0
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
                frame_count += 1
                expected_time = start_time + frame_count * frame_duration
                sleep_time = expected_time - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
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


@app.route("/api/history/image/records")
def api_history_image_records():
    history_type = request.args.get("type", "all")
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)
    per_page = min(per_page, 50)
    sort_by = request.args.get("sort_by", "uploaded_at")
    sort_order = request.args.get("sort_order", "desc")
    keyword = request.args.get("keyword", "")
    start_date = request.args.get("start_date", "")
    end_date = request.args.get("end_date", "")
    api_provider = request.args.get("api_provider", "")

    result = query_image_records(
        history_type=history_type,
        page=page,
        per_page=per_page,
        sort_by=sort_by,
        sort_order=sort_order,
        keyword=keyword,
        start_date=start_date,
        end_date=end_date,
        api_provider=api_provider,
    )
    return jsonify({"success": True, **result})


@app.route("/api/history/video/records")
def api_history_video_records():
    detection_target = request.args.get("target", "person")
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)
    per_page = min(per_page, 50)
    sort_by = request.args.get("sort_by", "uploaded_at")
    sort_order = request.args.get("sort_order", "desc")
    keyword = request.args.get("keyword", "")
    start_date = request.args.get("start_date", "")
    end_date = request.args.get("end_date", "")
    status_filter = request.args.get("status", "")

    result = query_video_records(
        detection_target=detection_target,
        page=page,
        per_page=per_page,
        sort_by=sort_by,
        sort_order=sort_order,
        keyword=keyword,
        start_date=start_date,
        end_date=end_date,
        status_filter=status_filter,
    )
    return jsonify({"success": True, **result})


@app.route("/api/history/image/detail/<int:record_id>")
def api_history_image_detail(record_id):
    record = db.session.get(ImageRecord, record_id)
    if not record:
        return jsonify({"success": False, "error": "记录不存在"}), 404
    return jsonify({"success": True, "record": serialize_image_record(record)})


@app.route("/api/history/video/detail/<int:record_id>")
def api_history_video_detail(record_id):
    record = db.session.get(VideoRecord, record_id)
    if not record:
        return jsonify({"success": False, "error": "记录不存在"}), 404
    return jsonify({"success": True, "record": serialize_video_record(record)})


@app.route("/api/history/image/batch-delete", methods=["POST"])
def api_history_image_batch_delete():
    data = request.get_json(silent=True) or {}
    ids = data.get("ids", [])
    if not ids:
        return jsonify({"success": False, "error": "未提供记录ID"}), 400

    deleted, errors = batch_delete_image_records(ids)
    return jsonify({
        "success": True,
        "deleted": deleted,
        "errors": errors,
        "message": f"成功删除 {deleted} 条记录",
    })


@app.route("/api/history/video/batch-delete", methods=["POST"])
def api_history_video_batch_delete():
    data = request.get_json(silent=True) or {}
    ids = data.get("ids", [])
    if not ids:
        return jsonify({"success": False, "error": "未提供记录ID"}), 400

    deleted, errors = batch_delete_video_records(ids)
    return jsonify({
        "success": True,
        "deleted": deleted,
        "errors": errors,
        "message": f"成功删除 {deleted} 条记录",
    })


@app.route("/api/export/start", methods=["POST"])
def api_export_start():
    data = request.get_json(silent=True) or {}
    export_type = data.get("type", "")
    params = {
        "format": data.get("format", "xlsx"),
        "fields": data.get("fields", []),
        "start_date": data.get("start_date", ""),
        "end_date": data.get("end_date", ""),
    }

    if export_type not in ("image_pedestrian", "image_vehicle", "video_pedestrian", "video_vehicle"):
        return jsonify({"success": False, "error": "无效的导出类型"}), 400

    task_id = create_export_task(export_type, params)

    thread = threading.Thread(target=run_export_task, args=(task_id, app), daemon=True)
    thread.start()

    return jsonify({"success": True, "task_id": task_id, "message": "导出任务已创建"})


@app.route("/api/export/progress/<task_id>")
def api_export_progress(task_id):
    task = get_task(task_id)
    if not task:
        return jsonify({"success": False, "error": "任务不存在"}), 404

    return jsonify({
        "success": True,
        "task": {
            "id": task["id"],
            "type": task["type"],
            "status": task["status"],
            "progress": task["progress"],
            "message": task["message"],
            "total_records": task["total_records"],
            "exported_records": task["exported_records"],
            "file_size": task["file_size"],
            "file_name": task["file_name"],
            "created_at": task["created_at"].strftime("%Y-%m-%d %H:%M:%S") if task["created_at"] else "",
            "completed_at": task["completed_at"].strftime("%Y-%m-%d %H:%M:%S") if task["completed_at"] else "",
            "error": task["error"],
        }
    })


@app.route("/api/export/download/<task_id>")
def api_export_download(task_id):
    task = get_task(task_id)
    if not task:
        return jsonify({"success": False, "error": "任务不存在"}), 404
    if task["status"] != "completed":
        return jsonify({"success": False, "error": "任务尚未完成"}), 400

    buffer = task.get("file_path")
    if not buffer:
        return jsonify({"success": False, "error": "导出文件不存在"}), 404

    format_map = {
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "csv": "text/csv",
        "json": "application/json",
    }
    file_format = task["params"].get("format", "xlsx")
    mimetype = format_map.get(file_format, "application/octet-stream")

    return send_file(
        buffer,
        as_attachment=True,
        download_name=task["file_name"],
        mimetype=mimetype,
    )


@app.route("/api/export/history")
def api_export_history():
    history = get_export_history(20)
    return jsonify({
        "success": True,
        "history": [{
            "id": t["id"],
            "type": t["type"],
            "status": t["status"],
            "progress": t["progress"],
            "message": t["message"],
            "total_records": t["total_records"],
            "exported_records": t["exported_records"],
            "file_name": t["file_name"],
            "file_size": t["file_size"],
            "created_at": t["created_at"].strftime("%Y-%m-%d %H:%M:%S") if t["created_at"] else "",
            "completed_at": t["completed_at"].strftime("%Y-%m-%d %H:%M:%S") if t["completed_at"] else "",
            "error": t["error"],
        } for t in history]
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
