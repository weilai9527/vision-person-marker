from datetime import datetime, timedelta

from sqlalchemy import func, and_

from models import db, DetectionResult, ImageRecord, VideoRecord


def _format_change(current: int, previous: int) -> tuple[str, str]:
    if previous <= 0:
        if current > 0:
            return "↑ 今日新增", "up"
        return "→ 暂无变化", ""
    diff = current - previous
    pct = round(abs(diff) / previous * 100)
    if diff > 0:
        return f"↑ 较昨日 +{pct}%", "up"
    if diff < 0:
        return f"↓ 较昨日 -{pct}%", "down"
    return "→ 与昨日持平", ""


def build_dashboard_stats() -> dict:
    now = datetime.utcnow()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_start = today_start - timedelta(days=1)
    week_start = today_start - timedelta(days=6)

    image_total = db.session.query(func.count(ImageRecord.id)).scalar() or 0
    video_total = db.session.query(func.count(VideoRecord.id)).scalar() or 0

    person_provider_conditions = and_(
        ~DetectionResult.llm_api_provider.like('%vehicle%'),
        DetectionResult.llm_api_provider.notin_(['local_yolo_vehicle', 'vehicle_yolo_judge', 'vision_model_vehicle_judge'])
    )

    today_person = db.session.query(
        func.coalesce(func.sum(DetectionResult.person_count), 0)
    ).filter(
        person_provider_conditions,
        DetectionResult.detected_at >= today_start
    ).scalar() or 0

    yesterday_person = db.session.query(
        func.coalesce(func.sum(DetectionResult.person_count), 0)
    ).filter(
        person_provider_conditions,
        DetectionResult.detected_at >= yesterday_start,
        DetectionResult.detected_at < today_start
    ).scalar() or 0

    person_change_text, person_change_class = _format_change(today_person, yesterday_person)

    trend_labels = []
    trend_values = []
    for offset in range(7):
        day = week_start + timedelta(days=offset)
        next_day = day + timedelta(days=1)
        trend_labels.append(day.strftime("%m-%d"))
        image_count = db.session.query(func.count(ImageRecord.id)).filter(
            ImageRecord.uploaded_at >= day,
            ImageRecord.uploaded_at < next_day
        ).scalar() or 0
        video_count = db.session.query(func.count(VideoRecord.id)).filter(
            VideoRecord.uploaded_at >= day,
            VideoRecord.uploaded_at < next_day
        ).scalar() or 0
        trend_values.append(image_count + video_count)

    local_count = db.session.query(func.count(DetectionResult.id)).filter(
        DetectionResult.llm_api_provider.like('%yolo%') |
        DetectionResult.llm_api_provider.like('local%')
    ).scalar() or 0

    total_detections = db.session.query(func.count(DetectionResult.id)).scalar() or 0
    ai_count = total_detections - local_count

    unprocessed_images = db.session.query(func.count(ImageRecord.id)).filter(
        ~ImageRecord.detections.any()
    ).scalar() or 0

    unprocessed_videos = db.session.query(func.count(VideoRecord.id)).filter(
        VideoRecord.status.notin_(['completed', 'done', 'success'])
    ).scalar() or 0
    unprocessed_count = unprocessed_images + unprocessed_videos

    person_image_total = db.session.query(
        func.coalesce(func.sum(DetectionResult.person_count), 0)
    ).filter(person_provider_conditions).scalar() or 0

    vehicle_image_total = db.session.query(
        func.coalesce(func.sum(DetectionResult.person_count), 0)
    ).filter(
        DetectionResult.llm_api_provider.in_(['local_yolo_vehicle', 'vehicle_yolo_judge', 'vision_model_vehicle_judge'])
    ).scalar() or 0

    person_video_total = db.session.query(
        func.coalesce(func.sum(
            func.coalesce(VideoRecord.unique_count, VideoRecord.total_persons, 0)
        ), 0)
    ).filter(VideoRecord.detection_target == 'person').scalar() or 0

    vehicle_video_total = db.session.query(
        func.coalesce(func.sum(
            func.coalesce(VideoRecord.unique_count, VideoRecord.total_persons, 0)
        ), 0)
    ).filter(VideoRecord.detection_target == 'vehicle').scalar() or 0

    video_completed = db.session.query(func.count(VideoRecord.id)).filter(
        VideoRecord.status.in_(['completed', 'done', 'success'])
    ).scalar() or 0

    video_processing = db.session.query(func.count(VideoRecord.id)).filter(
        VideoRecord.status.in_(['pending', 'processing'])
    ).scalar() or 0

    return {
        "today_person": today_person,
        "person_change_text": person_change_text,
        "person_change_class": person_change_class,
        "video_value": f"{video_completed}/{video_total}",
        "video_status": f"处理中 {video_processing} 个" if video_processing else "暂无处理任务",
        "camera_value": "待启动",
        "camera_status": "点击摄像头入口连接",
        "system_value": "正常",
        "system_status": f"累计 {image_total + video_total} 条记录",
        "trend_labels": trend_labels,
        "trend_values": trend_values,
        "method_labels": ["本地 YOLO", "AI 模型", "未处理"],
        "method_values": [local_count, ai_count, unprocessed_count],
        "type_labels": ["图片人员", "图片车辆", "视频人员", "视频车辆"],
        "type_values": [person_image_total, vehicle_image_total, person_video_total, vehicle_video_total],
        "confidence_labels": ["高置信度", "中等置信度", "低置信度"],
        "confidence_values": [0, total_detections, 0],
        "hourly_labels": ["0-6", "6-9", "9-12", "12-14", "14-18", "18-24"],
        "hourly_values": [0, 0, 0, 0, 0, 0],
    }
