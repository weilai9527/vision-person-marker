from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class ImageRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    original_filename = db.Column(db.String(255), nullable=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    original_image_path = db.Column(db.String(512), nullable=False)
    model_image_path = db.Column(db.String(512), nullable=True)
    original_width = db.Column(db.Integer)
    original_height = db.Column(db.Integer)
    model_width = db.Column(db.Integer)
    model_height = db.Column(db.Integer)

    detections = db.relationship('DetectionResult', backref='image', lazy=True, cascade="all, delete-orphan")

    def __repr__(self):
        return f'<ImageRecord {self.id} - {self.original_filename}>'


class DetectionResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_id = db.Column(db.Integer, db.ForeignKey('image_record.id'), nullable=False, index=True)
    detected_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    person_count = db.Column(db.Integer)
    bounding_boxes_json = db.Column(db.Text)
    llm_analysis_text = db.Column(db.Text)
    result_image_path = db.Column(db.String(512))
    llm_api_provider = db.Column(db.String(50), index=True)
    llm_model_name = db.Column(db.String(50))
    raw_llm_response_log_path = db.Column(db.String(512))
    raw_yolo_boxes_json = db.Column(db.Text, nullable=True)
    llm_boxes_json = db.Column(db.Text, nullable=True)
    final_source = db.Column(db.String(50), nullable=True, index=True)
    review_status = db.Column(db.String(20), default='pending', index=True)
    detection_strategy = db.Column(db.String(50), nullable=True)
    yolo_miss_reason = db.Column(db.String(100), nullable=True)

    def __repr__(self):
        return f'<DetectionResult {self.id} - {self.person_count} people>'


class VideoRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    original_filename = db.Column(db.String(255), nullable=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    detection_target = db.Column(db.String(20), default='person', nullable=False, index=True)
    yolo_model_name = db.Column(db.String(100), default='', nullable=False)
    video_path = db.Column(db.String(512), nullable=False)
    processed_video_path = db.Column(db.String(512), nullable=True)
    status = db.Column(db.String(20), default='pending', index=True)
    total_frames = db.Column(db.Integer, default=0)
    processed_frames = db.Column(db.Integer, default=0)
    fps = db.Column(db.Float, default=0.0)
    duration = db.Column(db.Float, default=0.0)
    total_persons = db.Column(db.Integer, default=0)
    unique_count = db.Column(db.Integer, default=0)
    sum_count = db.Column(db.Integer, default=0)
    avg_confidence = db.Column(db.Float, default=0.0)
    video_width = db.Column(db.Integer, default=0)
    video_height = db.Column(db.Integer, default=0)
    error_message = db.Column(db.Text, nullable=True)

    def __repr__(self):
        return f'<VideoRecord {self.id} - {self.original_filename}>'
