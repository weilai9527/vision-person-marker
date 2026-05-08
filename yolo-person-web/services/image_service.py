import base64
import json
import mimetypes
from datetime import datetime
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageDraw, ImageOps

from config import (
    LOG_DIR,
    MODEL_IMAGE_JPEG_QUALITY,
    MODEL_IMAGE_MAX_PIXELS,
    MODEL_IMAGE_MAX_SIDE,
    MODEL_IMAGE_MIN_SIDE,
    RESULT_DIR,
)
from services.detection_core import expand_box, infer_box_coordinate_size, normalize_box


def image_to_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    image_base64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{image_base64}"


def choose_model_image_size(width: int, height: int) -> tuple[int, int, int]:
    if MODEL_IMAGE_MAX_SIDE > 0:
        max_side = MODEL_IMAGE_MAX_SIDE
    else:
        total_pixels = width * height
        if total_pixels > 12_000_000:
            max_side = 1200
        elif total_pixels > 6_000_000:
            max_side = 1400
        elif total_pixels > 2_500_000:
            max_side = 1600
        else:
            max_side = 1800

    scale_by_side = min(1.0, max_side / max(width, height))
    scale_by_pixels = min(1.0, (MODEL_IMAGE_MAX_PIXELS / (width * height)) ** 0.5)
    scale = min(scale_by_side, scale_by_pixels)

    if scale < 1.0:
        target_width = max(1, round(width * scale))
        target_height = max(1, round(height * scale))
        if min(target_width, target_height) < MODEL_IMAGE_MIN_SIDE and min(width, height) >= MODEL_IMAGE_MIN_SIDE:
            min_side_scale = MODEL_IMAGE_MIN_SIDE / min(width, height)
            if min_side_scale < scale_by_side:
                target_width = max(1, round(width * min_side_scale))
                target_height = max(1, round(height * min_side_scale))
    else:
        target_width, target_height = width, height

    return target_width, target_height, max_side


def pil_image_to_data_url(image: Image.Image) -> tuple[str, int]:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=MODEL_IMAGE_JPEG_QUALITY, optimize=True)
    image_base64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{image_base64}", buffer.tell()


def prepare_model_image_object(image_path: Path) -> tuple[Image.Image, dict]:
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        original_width, original_height = image.size
        target_width, target_height, max_side = choose_model_image_size(original_width, original_height)
        if (target_width, target_height) != image.size:
            image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        return image.copy(), {
            "original_width": original_width,
            "original_height": original_height,
            "model_width": image.width,
            "model_height": image.height,
            "adaptive_max_side": max_side,
        }


def prepare_model_image(image_path: Path) -> tuple[str, int, int, dict]:
    image, image_info = prepare_model_image_object(image_path)
    data_url, jpeg_bytes = pil_image_to_data_url(image)
    image_info = {**image_info, "jpeg_bytes": jpeg_bytes}
    return data_url, image.width, image.height, image_info


def save_model_response_log(
    image_path: Path,
    api_config: dict,
    response_data: dict,
    content: str,
    image_info: dict,
) -> str:
    log_name = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}.json"
    log_path = LOG_DIR / log_name
    log_payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "provider": api_config.get("provider"),
        "api_url": api_config.get("api_url"),
        "model": api_config.get("model"),
        "image": image_path.name,
        "image_info": image_info,
        "raw_content": content,
        "raw_response": response_data,
    }
    log_path.write_text(json.dumps(log_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(log_path)


def draw_person_boxes(image_path: Path, boxes: list, box_image_size: tuple[int, int] | None = None) -> tuple[str, int]:
    result_name = f"{uuid4().hex}{image_path.suffix.lower()}"
    result_path = RESULT_DIR / result_name

    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        default_box_width, default_box_height = box_image_size or (width, height)
        box_width, box_height = infer_box_coordinate_size(boxes, default_box_width, default_box_height)
        scale_x = width / box_width
        scale_y = height / box_height
        draw = ImageDraw.Draw(image)
        line_width = max(3, min(width, height) // 160)
        drawn_count = 0

        for index, box in enumerate(boxes, start=1):
            normalized_box = normalize_box(box, box_width, box_height)
            if normalized_box is None:
                continue

            left, top, right, bottom = normalized_box
            left = max(0, min(width - 1, round(left * scale_x)))
            top = max(0, min(height - 1, round(top * scale_y)))
            right = max(0, min(width - 1, round(right * scale_x)))
            bottom = max(0, min(height - 1, round(bottom * scale_y)))
            left, top, right, bottom = expand_box((left, top, right, bottom), width, height)
            draw.rectangle((left, top, right, bottom), outline="#16a34a", width=line_width)
            drawn_count += 1

            label = str(index)
            text_box = draw.textbbox((0, 0), label)
            label_width = text_box[2] - text_box[0] + 12
            label_height = text_box[3] - text_box[1] + 8
            label_top = max(0, top - label_height)
            draw.rectangle((left, label_top, left + label_width, label_top + label_height), fill="#16a34a")
            draw.text((left + 6, label_top + 4), label, fill="#ffffff")

        image.save(result_path)

    return result_name, drawn_count
