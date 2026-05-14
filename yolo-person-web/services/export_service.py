import csv
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from io import BytesIO, StringIO
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Font, Alignment
from flask import jsonify, send_file

from models import db, ImageRecord, DetectionResult, VideoRecord

logger = logging.getLogger(__name__)

EXPORT_TASKS = {}

EXPORT_FORMATS = {
    "csv": {"label": "CSV", "mime": "text/csv"},
    "xlsx": {"label": "Excel", "mime": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"},
    "json": {"label": "JSON", "mime": "application/json"},
}

IMAGE_FIELD_MAP = {
    "record_id": "记录ID",
    "original_filename": "原始文件名",
    "uploaded_at": "上传时间",
    "original_size": "原始尺寸",
    "model_size": "模型尺寸",
    "detection_id": "检测ID",
    "detected_at": "检测时间",
    "detection_count": "检测数量",
    "model_description": "模型说明",
    "api_provider": "API提供商",
    "model_name": "模型名称",
}

VIDEO_FIELD_MAP = {
    "record_id": "记录ID",
    "original_filename": "原始文件名",
    "uploaded_at": "上传时间",
    "detection_target": "检测目标",
    "status": "状态",
    "total_frames": "总帧数",
    "processed_frames": "处理帧数",
    "video_size": "视频尺寸",
    "fps": "FPS",
    "duration": "时长(秒)",
    "max_count": "最大数量",
    "unique_count": "去重后总数",
    "avg_confidence": "平均置信度",
    "error_message": "错误信息",
}


def create_export_task(task_type, params):
    task_id = uuid.uuid4().hex[:12]
    EXPORT_TASKS[task_id] = {
        "id": task_id,
        "type": task_type,
        "params": params,
        "status": "pending",
        "progress": 0,
        "message": "等待处理...",
        "created_at": datetime.utcnow(),
        "completed_at": None,
        "file_path": None,
        "file_name": None,
        "file_size": 0,
        "total_records": 0,
        "exported_records": 0,
        "error": None,
    }
    return task_id


def update_task_progress(task_id, **kwargs):
    task = EXPORT_TASKS.get(task_id)
    if task:
        task.update(kwargs)


def get_task(task_id):
    return EXPORT_TASKS.get(task_id)


def get_export_history(limit=20):
    tasks = sorted(EXPORT_TASKS.values(), key=lambda t: t["created_at"], reverse=True)
    return tasks[:limit]


def build_image_query(detection_provider_filter=None, start_date=None, end_date=None):
    query = ImageRecord.query

    if start_date:
        query = query.filter(ImageRecord.uploaded_at >= start_date)
    if end_date:
        query = query.filter(ImageRecord.uploaded_at <= end_date)

    if detection_provider_filter:
        query = query.join(DetectionResult).filter(
            DetectionResult.llm_api_provider == detection_provider_filter
        )

    return query.order_by(ImageRecord.uploaded_at.desc())


def build_video_query(detection_target=None, start_date=None, end_date=None):
    query = VideoRecord.query

    if detection_target:
        query = query.filter(VideoRecord.detection_target == detection_target)
    if start_date:
        query = query.filter(VideoRecord.uploaded_at >= start_date)
    if end_date:
        query = query.filter(VideoRecord.uploaded_at <= end_date)

    return query.order_by(VideoRecord.uploaded_at.desc())


def run_export_task(task_id, app=None):
    task = EXPORT_TASKS.get(task_id)
    if not task:
        return

    task["status"] = "processing"
    task["message"] = "正在查询数据..."
    task["progress"] = 5

    ctx = app.app_context() if app else None
    if ctx:
        ctx.push()

    try:
        params = task["params"]
        export_type = task["type"]
        file_format = params.get("format", "xlsx")
        fields = params.get("fields", [])
        start_date = params.get("start_date")
        end_date = params.get("end_date")

        if start_date:
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if end_date:
            end_date = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)

        if export_type == "image_pedestrian":
            records = build_image_query(
                detection_provider_filter=None,
                start_date=start_date,
                end_date=end_date,
            ).all()
            file_prefix = "图片行人检测数据"
            is_image = True
            is_pedestrian = True
        elif export_type == "image_vehicle":
            records = build_image_query(
                detection_provider_filter="local_yolo_vehicle",
                start_date=start_date,
                end_date=end_date,
            ).all()
            file_prefix = "图片车辆检测数据"
            is_image = True
            is_pedestrian = False
        elif export_type == "video_pedestrian":
            records = build_video_query(
                detection_target="person",
                start_date=start_date,
                end_date=end_date,
            ).all()
            file_prefix = "视频行人检测数据"
            is_image = False
            is_pedestrian = True
        elif export_type == "video_vehicle":
            records = build_video_query(
                detection_target="vehicle",
                start_date=start_date,
                end_date=end_date,
            ).all()
            file_prefix = "视频车辆检测数据"
            is_image = False
            is_pedestrian = False
        else:
            raise ValueError(f"Unknown export type: {export_type}")

        task["total_records"] = len(records)
        task["message"] = f"找到 {len(records)} 条记录，正在生成导出文件..."
        task["progress"] = 20

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{file_prefix}_{timestamp}.{file_format}"

        if file_format == "xlsx":
            output, exported = export_to_excel(records, fields, is_image, is_pedestrian, task)
        elif file_format == "csv":
            output, exported = export_to_csv(records, fields, is_image, is_pedestrian, task)
        elif file_format == "json":
            output, exported = export_to_json(records, fields, is_image, is_pedestrian, task)
        else:
            raise ValueError(f"Unsupported format: {file_format}")

        task["exported_records"] = exported
        task["progress"] = 100
        task["status"] = "completed"
        task["message"] = f"导出完成，共导出 {exported} 条记录"
        task["completed_at"] = datetime.utcnow()
        task["file_path"] = output
        task["file_name"] = file_name
        if hasattr(output, 'tell'):
            output.seek(0, 2)
            task["file_size"] = output.tell()
            output.seek(0)

    except Exception as exc:
        logger.exception("Export task failed")
        task["status"] = "failed"
        task["message"] = f"导出失败: {exc}"
        task["error"] = str(exc)
        task["progress"] = 0
    finally:
        if ctx:
            ctx.pop()


def get_image_row_data(record, fields, is_pedestrian=True):
    rows = []
    if record.detections:
        detections_to_use = record.detections
        if not is_pedestrian:
            detections_to_use = [d for d in record.detections if d.llm_api_provider == "local_yolo_vehicle"]
        for detection in (detections_to_use if detections_to_use else record.detections):
            row = {}
            for field in fields:
                if field == "record_id":
                    row["记录ID"] = record.id
                elif field == "original_filename":
                    row["原始文件名"] = record.original_filename
                elif field == "uploaded_at":
                    row["上传时间"] = record.uploaded_at.strftime("%Y-%m-%d %H:%M:%S") if record.uploaded_at else ""
                elif field == "original_size":
                    row["原始尺寸"] = f"{record.original_width or '-'}x{record.original_height or '-'}"
                elif field == "model_size":
                    row["模型尺寸"] = f"{record.model_width or '-'}x{record.model_height or '-'}"
                elif field == "detection_id":
                    row["检测ID"] = detection.id
                elif field == "detected_at":
                    row["检测时间"] = detection.detected_at.strftime("%Y-%m-%d %H:%M:%S") if detection.detected_at else ""
                elif field == "detection_count":
                    row["检测数量"] = detection.person_count
                elif field == "model_description":
                    row["模型说明"] = detection.llm_analysis_text or ""
                elif field == "api_provider":
                    row["API提供商"] = detection.llm_api_provider or ""
                elif field == "model_name":
                    row["模型名称"] = detection.llm_model_name or ""
            rows.append(row)
    else:
        row = {}
        for field in fields:
            if field == "record_id":
                row["记录ID"] = record.id
            elif field == "original_filename":
                row["原始文件名"] = record.original_filename
            elif field == "uploaded_at":
                row["上传时间"] = record.uploaded_at.strftime("%Y-%m-%d %H:%M:%S") if record.uploaded_at else ""
            elif field == "original_size":
                row["原始尺寸"] = f"{record.original_width or '-'}x{record.original_height or '-'}"
            elif field == "model_size":
                row["模型尺寸"] = f"{record.model_width or '-'}x{record.model_height or '-'}"
            elif field in ("detection_id", "detected_at", "detection_count", "model_description", "api_provider", "model_name"):
                row[IMAGE_FIELD_MAP[field]] = ""
        rows.append(row)
    return rows


def get_video_row_data(record, fields):
    row = {}
    for field in fields:
        if field == "record_id":
            row["记录ID"] = record.id
        elif field == "original_filename":
            row["原始文件名"] = record.original_filename
        elif field == "uploaded_at":
            row["上传时间"] = record.uploaded_at.strftime("%Y-%m-%d %H:%M:%S") if record.uploaded_at else ""
        elif field == "detection_target":
            row["检测目标"] = record.detection_target or "person"
        elif field == "status":
            row["状态"] = record.status or ""
        elif field == "total_frames":
            row["总帧数"] = record.total_frames
        elif field == "processed_frames":
            row["处理帧数"] = record.processed_frames
        elif field == "video_size":
            row["视频尺寸"] = f"{record.video_width or '-'}x{record.video_height or '-'}"
        elif field == "fps":
            row["FPS"] = round(record.fps, 2) if record.fps else ""
        elif field == "duration":
            row["时长(秒)"] = round(record.duration, 2) if record.duration else ""
        elif field == "max_count":
            row["最大数量"] = record.total_persons
        elif field == "unique_count":
            row["去重后总数"] = record.unique_count or record.total_persons or 0
        elif field == "avg_confidence":
            row["平均置信度"] = round(record.avg_confidence, 4) if record.avg_confidence else ""
        elif field == "error_message":
            row["错误信息"] = record.error_message or ""
    return [row]


def export_to_excel(records, fields, is_image, is_pedestrian, task):
    wb = Workbook()
    ws = wb.active
    ws.title = "检测数据"

    if is_image:
        all_fields = fields if fields else list(IMAGE_FIELD_MAP.keys())
        headers = [IMAGE_FIELD_MAP[f] for f in all_fields if f in IMAGE_FIELD_MAP]
    else:
        all_fields = fields if fields else list(VIDEO_FIELD_MAP.keys())
        headers = [VIDEO_FIELD_MAP[f] for f in all_fields if f in VIDEO_FIELD_MAP]

    ws.append(headers)
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center")

    exported = 0
    total = len(records)
    for i, record in enumerate(records):
        if is_image:
            rows = get_image_row_data(record, all_fields, is_pedestrian)
        else:
            rows = get_video_row_data(record, all_fields)

        for row_data in rows:
            ws.append([row_data.get(h, "") for h in headers])
            exported += 1

        progress = 20 + int(60 * (i + 1) / total) if total > 0 else 80
        task["progress"] = min(progress, 90)
        task["message"] = f"正在处理第 {i + 1}/{total} 条记录..."

    for i, col in enumerate(ws.columns, 1):
        max_len = 0
        for cell in col:
            if cell.value:
                max_len = max(max_len, len(str(cell.value)))
        ws.column_dimensions[chr(64 + i) if i <= 26 else 'A'].width = min(max_len + 4, 60)

    buffer = BytesIO()
    wb.save(buffer)
    buffer.seek(0)
    return buffer, exported


def export_to_csv(records, fields, is_image, is_pedestrian, task):
    output = StringIO()

    if is_image:
        all_fields = fields if fields else list(IMAGE_FIELD_MAP.keys())
        headers = [IMAGE_FIELD_MAP[f] for f in all_fields if f in IMAGE_FIELD_MAP]
    else:
        all_fields = fields if fields else list(VIDEO_FIELD_MAP.keys())
        headers = [VIDEO_FIELD_MAP[f] for f in all_fields if f in VIDEO_FIELD_MAP]

    if hasattr(output, 'write'):
        import sys
        if sys.version_info[0] >= 3:
            output.write('\ufeff')

    writer = csv.writer(output)
    writer.writerow(headers)

    exported = 0
    total = len(records)
    for i, record in enumerate(records):
        if is_image:
            rows = get_image_row_data(record, all_fields, is_pedestrian)
        else:
            rows = get_video_row_data(record, all_fields)

        for row_data in rows:
            writer.writerow([row_data.get(h, "") for h in headers])
            exported += 1

        progress = 20 + int(60 * (i + 1) / total) if total > 0 else 80
        task["progress"] = min(progress, 90)
        task["message"] = f"正在处理第 {i + 1}/{total} 条记录..."

    buffer = BytesIO()
    buffer.write(output.getvalue().encode('utf-8-sig'))
    buffer.seek(0)
    return buffer, exported


def export_to_json(records, fields, is_image, is_pedestrian, task):
    all_records_data = []
    total = len(records)
    for i, record in enumerate(records):
        if is_image:
            all_fields = fields if fields else list(IMAGE_FIELD_MAP.keys())
            rows = get_image_row_data(record, all_fields, is_pedestrian)
        else:
            all_fields = fields if fields else list(VIDEO_FIELD_MAP.keys())
            rows = get_video_row_data(record, all_fields)

        all_records_data.extend(rows)

        progress = 20 + int(60 * (i + 1) / total) if total > 0 else 80
        task["progress"] = min(progress, 90)
        task["message"] = f"正在处理第 {i + 1}/{total} 条记录..."

    json_output = json.dumps(all_records_data, ensure_ascii=False, indent=2, default=str)
    buffer = BytesIO()
    buffer.write(json_output.encode('utf-8'))
    buffer.seek(0)
    return buffer, len(all_records_data)
