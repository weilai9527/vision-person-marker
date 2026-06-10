import logging
import time

from flask import Flask, jsonify, render_template, request
from PIL import Image

from config import CAMERA_CONF, CAMERA_IMGSZ, CAMERA_IOU, CAMERA_MIN_AREA
from yolo_detector import PERSON_CLASS_IDS, VEHICLE_CLASS_IDS, run_detection_on_image


app = Flask(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({"success": True, "status": "ok"})


@app.post("/api/detect")
def detect_frame():
    frame = request.files.get("frame")
    if frame is None or frame.filename == "":
        return jsonify({"success": False, "error": "没有收到摄像头画面"}), 400

    target = request.form.get("target", "person")
    if target == "vehicle":
        class_ids = VEHICLE_CLASS_IDS
        target_label = "vehicle"
    else:
        class_ids = PERSON_CLASS_IDS
        target_label = "person"

    started_at = time.perf_counter()
    try:
        with Image.open(frame.stream) as image:
            boxes, width, height = run_detection_on_image(
                image,
                class_ids=class_ids,
                target_label=target_label,
                conf=CAMERA_CONF,
                iou=CAMERA_IOU,
                imgsz=CAMERA_IMGSZ,
                min_area=CAMERA_MIN_AREA,
            )
    except Exception as exc:
        logger.exception("Camera detection failed")
        return jsonify({"success": False, "error": str(exc)}), 500

    elapsed_ms = round((time.perf_counter() - started_at) * 1000)
    return jsonify(
        {
            "success": True,
            "target": target_label,
            "count": len(boxes),
            "boxes": boxes,
            "width": width,
            "height": height,
            "elapsed_ms": elapsed_ms,
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False, threaded=True)
