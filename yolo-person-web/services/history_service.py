import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from models import db, ImageRecord, DetectionResult, VideoRecord

logger = logging.getLogger(__name__)

IMAGE_FIELD_LABELS = {
    "id": "记录ID",
    "original_filename": "文件名",
    "uploaded_at": "上传时间",
    "original_size": "原始尺寸",
    "model_size": "模型尺寸",
    "detection_count": "检测数量",
    "api_provider": "检测方式",
    "model_name": "模型名称",
    "detected_at": "检测时间",
}

VIDEO_FIELD_LABELS = {
    "id": "记录ID",
    "original_filename": "文件名",
    "uploaded_at": "上传时间",
    "detection_target": "检测目标",
    "status": "处理状态",
    "total_frames": "总帧数",
    "processed_frames": "已处理帧数",
    "fps": "FPS",
    "duration": "时长(秒)",
    "max_count": "最大数量",
    "unique_count": "去重总数",
    "avg_confidence": "平均置信度",
}


def serialize_image_record(record):
    detections = []
    for d in (record.detections or []):
        detections.append({
            "id": d.id,
            "detected_at": d.detected_at.strftime("%Y-%m-%d %H:%M:%S") if d.detected_at else "",
            "person_count": d.person_count,
            "api_provider": d.llm_api_provider or "",
            "model_name": d.llm_model_name or "",
            "analysis_text": d.llm_analysis_text or "",
            "result_image_path": d.result_image_path.replace("\\", "/") if d.result_image_path else "",
            "bounding_boxes": (json.loads(d.bounding_boxes_json) if d.bounding_boxes_json else []),
        })

    return {
        "id": record.id,
        "original_filename": record.original_filename,
        "uploaded_at": record.uploaded_at.strftime("%Y-%m-%d %H:%M:%S") if record.uploaded_at else "",
        "original_width": record.original_width,
        "original_height": record.original_height,
        "model_width": record.model_width,
        "model_height": record.model_height,
        "original_image_path": record.original_image_path.replace("\\", "/") if record.original_image_path else "",
        "detection_count": len(detections),
        "detections": detections,
    }


def serialize_video_record(record):
    return {
        "id": record.id,
        "original_filename": record.original_filename,
        "uploaded_at": record.uploaded_at.strftime("%Y-%m-%d %H:%M:%S") if record.uploaded_at else "",
        "detection_target": record.detection_target,
        "status": record.status,
        "total_frames": record.total_frames,
        "processed_frames": record.processed_frames,
        "fps": round(record.fps, 2) if record.fps else 0,
        "duration": round(record.duration, 2) if record.duration else 0,
        "total_persons": record.total_persons,
        "unique_count": record.unique_count,
        "sum_count": record.sum_count,
        "avg_confidence": round(record.avg_confidence, 4) if record.avg_confidence else 0,
        "video_width": record.video_width,
        "video_height": record.video_height,
        "video_path": record.video_path.replace("\\", "/") if record.video_path else "",
        "processed_video_path": record.processed_video_path.replace("\\", "/") if record.processed_video_path else "",
        "error_message": record.error_message or "",
        "has_result": bool(record.processed_video_path),
    }


def query_image_records(history_type="all", page=1, per_page=20,
                        sort_by="uploaded_at", sort_order="desc",
                        keyword="", start_date="", end_date="",
                        api_provider=""):
    query = ImageRecord.query

    if history_type == "pedestrian":
        query = query.filter(ImageRecord.detections.any(
            DetectionResult.llm_api_provider != "local_yolo_vehicle"
        ))
    elif history_type == "vehicle":
        query = query.join(DetectionResult).filter(
            DetectionResult.llm_api_provider == "local_yolo_vehicle"
        ).distinct()

    if keyword:
        like = f"%{keyword}%"
        query = query.filter(ImageRecord.original_filename.ilike(like))

    if start_date:
        try:
            sd = datetime.strptime(start_date, "%Y-%m-%d")
            query = query.filter(ImageRecord.uploaded_at >= sd)
        except ValueError:
            pass

    if end_date:
        try:
            ed = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
            query = query.filter(ImageRecord.uploaded_at < ed)
        except ValueError:
            pass

    if api_provider and api_provider != "all":
        if api_provider == "local_yolo":
            query = query.filter(ImageRecord.detections.any(
                DetectionResult.llm_api_provider == "local_yolo"
            ))
        elif api_provider == "local_yolo_vehicle":
            query = query.filter(ImageRecord.detections.any(
                DetectionResult.llm_api_provider == "local_yolo_vehicle"
            ))
        else:
            query = query.filter(ImageRecord.detections.any(
                DetectionResult.llm_api_provider == api_provider
            ))

    sort_col = getattr(ImageRecord, sort_by, ImageRecord.uploaded_at)
    if sort_order == "asc":
        query = query.order_by(sort_col.asc())
    else:
        query = query.order_by(sort_col.desc())

    pagination = query.paginate(page=page, per_page=per_page, error_out=False)

    return {
        "records": [serialize_image_record(r) for r in pagination.items],
        "page": pagination.page,
        "pages": pagination.pages,
        "total": pagination.total,
        "has_prev": pagination.has_prev,
        "has_next": pagination.has_next,
        "per_page": per_page,
    }


def query_video_records(detection_target="person", page=1, per_page=20,
                        sort_by="uploaded_at", sort_order="desc",
                        keyword="", start_date="", end_date="",
                        status_filter=""):
    query = VideoRecord.query.filter_by(detection_target=detection_target)

    if keyword:
        like = f"%{keyword}%"
        query = query.filter(VideoRecord.original_filename.ilike(like))

    if start_date:
        try:
            sd = datetime.strptime(start_date, "%Y-%m-%d")
            query = query.filter(VideoRecord.uploaded_at >= sd)
        except ValueError:
            pass

    if end_date:
        try:
            ed = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
            query = query.filter(VideoRecord.uploaded_at < ed)
        except ValueError:
            pass

    if status_filter and status_filter != "all":
        query = query.filter(VideoRecord.status == status_filter)

    sort_col = getattr(VideoRecord, sort_by, VideoRecord.uploaded_at)
    if sort_order == "asc":
        query = query.order_by(sort_col.asc())
    else:
        query = query.order_by(sort_col.desc())

    pagination = query.paginate(page=page, per_page=per_page, error_out=False)

    return {
        "records": [serialize_video_record(r) for r in pagination.items],
        "page": pagination.page,
        "pages": pagination.pages,
        "total": pagination.total,
        "has_prev": pagination.has_prev,
        "has_next": pagination.has_next,
        "per_page": per_page,
    }


def delete_image_record(record_id):
    record = db.session.get(ImageRecord, record_id)
    if not record:
        return False, "记录不存在"

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
        return True, "删除成功"
    except Exception as exc:
        db.session.rollback()
        logger.exception(f"删除图片记录 {record_id} 失败")
        return False, str(exc)


def batch_delete_image_records(ids):
    deleted = 0
    errors = []
    for rid in ids:
        success, msg = delete_image_record(rid)
        if success:
            deleted += 1
        else:
            errors.append({"id": rid, "error": msg})
    return deleted, errors


def delete_video_record(record_id):
    from services.video_service import get_video_count_metadata_path

    record = db.session.get(VideoRecord, record_id)
    if not record:
        return False, "记录不存在"

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
        return True, "删除成功"
    except Exception as exc:
        db.session.rollback()
        logger.exception(f"删除视频记录 {record_id} 失败")
        return False, str(exc)


def batch_delete_video_records(ids):
    deleted = 0
    errors = []
    for rid in ids:
        success, msg = delete_video_record(rid)
        if success:
            deleted += 1
        else:
            errors.append({"id": rid, "error": msg})
    return deleted, errors
